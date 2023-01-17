import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno



#sklearn imports

from sklearn.preprocessing import OrdinalEncoder

#imputer

from fancyimpute import KNN

from fancyimpute import IterativeImputer



#visulizations

import matplotlib.pyplot as plt

import seaborn as sns



#statistics

from statsmodels.stats.outliers_influence import variance_inflation_factor



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

local=False

if local is False:

    import os

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            print(os.path.join(dirname, filename))
if local == False:

    traindf=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

    testdf=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

else:

    traindf=pd.read_csv("train.csv")

    testdf=pd.read_csv("test.csv")

print(traindf.shape)

print(testdf.shape)
testdf.head()
training_ids = traindf["Id"]

testing_ids = testdf["Id"]

traindf_cp = traindf.copy()

testdf_cp = testdf.copy()

dependent_data = traindf[["Id","SalePrice"]]

traindf_cp.drop("SalePrice",axis=1,inplace=True)

print(traindf_cp.shape)

print(testdf_cp.shape)

if traindf_cp.shape[1]==testdf_cp.shape[1]:

    house_data = pd.concat([traindf_cp,testdf_cp],axis=0)

    house_data = house_data.reset_index(drop=True)

    house_data.fillna(np.nan,inplace=True)

    print(house_data.shape)
house_data.info()
def show_missing_info(house_data):

    missing_info = house_data.isna().sum().reset_index(drop=False)

    missing_info.columns = ["column","rows"]

    missing_info["missing_pct"] = (missing_info["rows"]/house_data.shape[0])*100

    missing_info = missing_info[missing_info["rows"]>0].sort_values(by="missing_pct",ascending=False)

    return missing_info

missing_df = show_missing_info(house_data)

missing_df
msno.bar(house_data,labels=house_data.columns.tolist())
msno.matrix(house_data,labels=house_data.columns.tolist())
delete_rows_cols = missing_df[missing_df["rows"]<20]["column"].tolist()

house_data.dropna(axis=0,how="any",subset=delete_rows_cols,inplace=True)

print(house_data.shape)
house_data.drop(columns=["PoolQC","MiscFeature","Alley"],axis=1,inplace=True)
house_data_columns = house_data.columns
categorical_columns = "Fence,FireplaceQu,GarageFinish,GarageQual,GarageCond,GarageType,BsmtExposure,BsmtCond,BsmtQual,BsmtFinType2,BsmtFinType1,MasVnrType".split(",")
def ordinal_encoding(data):

    #empty dictionary ordinal_enc_dict

    ordinal_enc_dict = {}

    for col_name in categorical_columns:

        # Create Ordinal encoder for col

        ordinal_enc_dict[col_name] = OrdinalEncoder()

        col = data[col_name]



        # Select non-null values of col

        col_not_null = col[col.notnull()]

        reshaped_vals = col_not_null.values.reshape(-1, 1)

        encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)



        # Store the values to non-null values of the column in users

        data.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)

    return data,ordinal_enc_dict
house_data_encoded,encoded_dict = ordinal_encoding(house_data.copy(deep=True))
house_data_encoded[categorical_columns].head()
for i in categorical_columns:

    print(house_data_encoded[i].unique())

    print("-"*40)
def impute_categorical_features(data,encoded_dict):

    # Create KNN imputer

    KNN_imputer = KNN()

    data.iloc[:, :] = np.round(KNN_imputer.fit_transform(data))

    for col_name in categorical_columns:

        reshaped = data[col_name].values.reshape(-1, 1)

        data[col_name] = encoded_dict[col_name].inverse_transform(reshaped)

    return data
house_data_imputed = impute_categorical_features(house_data_encoded[categorical_columns],encoded_dict)
house_data.drop(columns=categorical_columns,inplace=True,axis=1)

house_data = pd.concat([house_data,house_data_imputed],axis=1)

house_data = house_data[house_data_columns]
house_data[categorical_columns].head()
missing_df = show_missing_info(house_data)

missing_df
missing_columns = missing_df["column"].tolist()
tmpdf = house_data[missing_columns]

tmpdf_index = tmpdf.index
MICE_imputer = IterativeImputer()

house_data_mice = MICE_imputer.fit_transform(tmpdf)
imputed_df = pd.DataFrame(house_data_mice,columns=missing_columns,index=tmpdf_index)
house_data.drop(columns=missing_columns,inplace=True,axis=1)

house_data = pd.concat([house_data,imputed_df],axis=1)

print(house_data.shape)
missing_df = show_missing_info(house_data)

missing_df
pp_house_data_train = house_data[house_data["Id"].isin(training_ids.tolist())]

pp_house_data_test = house_data[house_data["Id"].isin(testing_ids.tolist())]
sales_price = dependent_data[dependent_data["Id"].isin(training_ids.tolist())]

pp_house_data_train.insert(0,"SalePrice",sales_price["SalePrice"])
pp_house_data_train.to_csv("pp_train.csv",index=False)

pp_house_data_test.to_csv("pp_test.csv",index=False)
train_df = pd.read_csv("pp_train.csv")

train_df.head()
area_features = "LotFrontage,LotArea,MasVnrArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,1stFlrSF,2ndFlrSF,LowQualFinSF,GrLivArea,GarageArea,WoodDeckSF,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch,PoolArea".split(",")

years_and_dates = "YearBuilt,YearRemodAdd,GarageYrBlt,MoSold,YrSold".split(",")

quantitiative_descrete_columns = "FullBath,HalfBath,BsmtFullBath,BsmtHalfBath,TotRmsAbvGrd,Fireplaces,GarageCars,MiscVal".split(",")
categorical_features = list(set(list(train_df.columns[2:])) - set(area_features+years_and_dates+quantitiative_descrete_columns))
for i in area_features+quantitiative_descrete_columns+years_and_dates:

    train_df[i] = train_df[i].astype(float)
train_df[years_and_dates].describe()
train_df[quantitiative_descrete_columns].describe()
corr = train_df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

fig, ax = plt.subplots(figsize=(20,15))

sns.heatmap(corr,mask=mask,center=0, linewidths=0, annot=True, fmt=".2f")
qc_correlation_with_saleprice = train_df[["SalePrice"]+area_features].corr()["SalePrice"][1:]

fig = plt.subplots(figsize=(16, 5))

plt.bar(qc_correlation_with_saleprice.index,qc_correlation_with_saleprice)

plt.xticks(rotation=90)

plt.title("quantitative continues features correlation with sales price")

plt.xlabel("category")

plt.ylabel("correlation")

plt.grid()

plt.show()
qd_correlation_with_saleprice = train_df[["SalePrice"]+quantitiative_descrete_columns].corr()["SalePrice"][1:]

fig = plt.subplots(figsize=(16, 5))

plt.bar(qd_correlation_with_saleprice.index,qd_correlation_with_saleprice)

plt.xticks(rotation=90)

plt.title("quantitiative descrete features correlation with sales price")

plt.xlabel("category")

plt.ylabel("correlation")

plt.grid()

plt.show()
cn_correlation_with_saleprice = train_df[["SalePrice"]+categorical_features].corr()["SalePrice"][1:]

fig = plt.subplots(figsize=(16, 5))

plt.bar(cn_correlation_with_saleprice.index,cn_correlation_with_saleprice)

plt.xticks(rotation=90)

plt.title("Categorical features correlation with sales price")

plt.xlabel("category")

plt.ylabel("correlation")

plt.grid()

plt.show()
fig,ax = plt.subplots(6,3,figsize=(15,15))

splits = np.split(np.array(area_features), 6)

for k,sp in enumerate(splits):

    for i,col in enumerate(sp):

        ax[k,i].scatter(train_df[col],train_df["SalePrice"])

        ax[k,i].set_title(col)



plt.tight_layout()

plt.show()
fig,ax = plt.subplots(4,2,figsize=(15,15))

splits = np.split(np.array(quantitiative_descrete_columns), 4)

for k,sp in enumerate(splits):

    for i,col in enumerate(sp):

        ax[k,i].scatter(train_df[col],train_df["SalePrice"])

        ax[k,i].set_title(col)

    

plt.tight_layout()

plt.show()
def plot_selected_columns_scatter(selected_columns):

    for i in selected_columns:

        fig,ax = plt.subplots(figsize=(15,4))

        plt.scatter(train_df[i],train_df["SalePrice"])

        plt.title(f"{i} vs Sale Price")

        plt.xlabel(i)

        plt.ylabel("Sale Price")

        plt.grid()

        plt.show()
selected_columns_qc = qc_correlation_with_saleprice[qc_correlation_with_saleprice>0.5].sort_values(ascending=False).index

selected_columns_qd = qd_correlation_with_saleprice[qd_correlation_with_saleprice>0.5].sort_values(ascending=False).index

plot_selected_columns_scatter(selected_columns_qc.tolist()+selected_columns_qd.tolist())
outliers = {"GrLivArea":{"sales_price":200000,"value":4000},

            "GarageArea":{"sales_price":300000,"value":1200},

            "TotalBsmtSF":{"sales_price":200000,"value":6000},

            "1stFlrSF":{"sales_price":200000,"value":4000}}

print(train_df.shape)

for col in selected_columns_qc:

    ol = outliers.get(col)

    train_df = train_df[~((train_df["SalePrice"] < ol["sales_price"]) & (train_df[col] > ol["value"]))]

print(train_df.shape)
plot_selected_columns_scatter(selected_columns_qc)
fig,ax = plt.subplots(figsize=(10,5))

sns.distplot(train_df["SalePrice"])
original_sale_prices = traindf["SalePrice"]

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

fig,ax = plt.subplots(figsize=(10,5))

sns.distplot(train_df["SalePrice"])
train_df.drop("Id",axis=1,inplace=True)
train_df = pd.get_dummies(train_df,columns=categorical_features,drop_first=True)

train_df = train_df.apply(pd.to_numeric)
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale
def prepare_data(data):

    X = data.drop("SalePrice",axis=1)

    y = data["SalePrice"]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

    return X_train,X_test,y_train,y_test
def fit_baseline_model(X_train,X_test,y_train,y_test):

    model = LinearRegression()

    model.fit(X_train,y_train)

    print(f"Training set R2 {model.score(X_train,y_train)}")

    y_pred = model.predict(X_test)

    print(f"R2 score {r2_score(y_test,y_pred)}")

    print(f"RMSE {np.sqrt(mean_squared_error(y_test,y_pred))}")

    return y_pred
def show_pred_and_test(y_test,y_pred):

    plt.figure(figsize=(20,5))

    plt.plot(y_test.values,label="Actual")

    plt.plot(y_pred,label="Predicted")

    plt.legend()

    plt.show()
X_train,X_test,y_train,y_test = prepare_data(train_df)

y_pred = fit_baseline_model(X_train,X_test,y_train,y_test)

show_pred_and_test(y_test,y_pred)
from sklearn.ensemble import BaggingRegressor
X_train,X_test,y_train,y_test = prepare_data(train_df)

bag_reg = BaggingRegressor(LinearRegression(),

                          n_estimators=200,

                          bootstrap=True,

                          max_samples=0.7,

                          n_jobs=-1,

                          oob_score=True)

bag_reg.fit(X_train,y_train)

print(f"Out of bag score {bag_reg.oob_score_}")

print(f"Training set R2 {bag_reg.score(X_train,y_train)}")

y_pred = bag_reg.predict(X_test)

print(f"R2 score {r2_score(y_test,y_pred)}")

print(f"RMSE {np.sqrt(mean_squared_error(y_test,y_pred))}")
show_pred_and_test(y_test,y_pred)