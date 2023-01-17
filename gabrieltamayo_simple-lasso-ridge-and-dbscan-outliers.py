# import packages

import pandas as pd

import numpy as np

from scipy.stats import skew

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import RobustScaler

from sklearn.linear_model import Ridge,RidgeCV, Lasso, ElasticNet, LassoCV, LassoLarsCV, ElasticNetCV

from sklearn.model_selection import cross_val_score

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# define plot styles

plt.style.use(['dark_background'])

params = {'legend.fontsize': 'x-large',

        'figure.figsize': (15, 5),

        'axes.labelsize': 'small',

        'axes.titlesize':'small',

        'xtick.labelsize':'small',

        'ytick.labelsize':'small'}

plt.rcParams.update(params)
# read train and test data

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train_shape = train.shape

test_shape = test.shape

print(f"Train data: rows {train_shape[0]} | columns {train_shape[1]}")

print(f"Test data: rows {test_shape[0]} | columns {test_shape[1]}")
diff_columns = [col for col in train.columns if col not in test.columns]

inters_columns = [col for col in train.columns if col in test.columns]

print(f"Differents columns between train and test ({len(diff_columns)}) {diff_columns}", end="\n"*2)

print(f"Intersect columns between train and test ({len(inters_columns)}) {inters_columns}")
# transform data type of MSSubClass, YrSold and MoSold to object in train and test



train['MSSubClass'] = train['MSSubClass'].apply(str)

train['YrSold'] = train['YrSold'].astype(str)

train['MoSold'] = train['MoSold'].astype(str)

train['GarageYrBlt'] = train['GarageYrBlt'].astype(str)





test['MSSubClass'] = test['MSSubClass'].apply(str)

test['YrSold'] = test['YrSold'].astype(str)

test['MoSold'] = test['MoSold'].astype(str)

test['GarageYrBlt'] = test['GarageYrBlt'].astype(str)
# remove outliers with percentile technique

def outliers_box_plot(data):

    q1, q3= np.percentile(data,[25,75])

    iqr = q3 - q1

    lower_bound = q1 -(1.5 * iqr) 

    upper_bound = q3 +(1.5 * iqr) 

    return lower_bound, upper_bound
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

ax = sns.boxplot(x="OverallQual", y="SalePrice", data=train);

ax = sns.swarmplot(x="OverallQual", y="SalePrice", data=train, color=".15")





# remove outliers in SalePrice for each group conditioned on OverallQual

drop_index = []

for g in train.groupby("OverallQual")["SalePrice"]:

    serie_group = g[1]

    l,u = outliers_box_plot(serie_group)

    if len(serie_group)>10: # arbitrarily defined        

        drop_index = drop_index + (serie_group[(serie_group<l) |( serie_group>u)].index.values.tolist())



print(f"Eliminated outlier rows: {len(drop_index)}")

train.drop(drop_index,inplace=True)





plt.subplot(1,2,2)

ax = sns.boxplot(x="OverallQual", y="SalePrice", data=train);

ax = sns.swarmplot(x="OverallQual", y="SalePrice", data=train, color=".15")
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

ax = sns.boxplot(x="MSZoning", y="SalePrice", data=train);

ax = sns.swarmplot(x="MSZoning", y="SalePrice", data=train, color=".15")





# remove outliers in SalePrice for each group conditioned on OverallQual

drop_index = []

for g in train.groupby("MSZoning")["SalePrice"]:

    serie_group = g[1]

    l,u = outliers_box_plot(serie_group)

    if len(serie_group)>10: # arbitrarily defined        

        drop_index = drop_index + (serie_group[(serie_group<l) |( serie_group>u)].index.values.tolist())



print(f"Eliminated outlier rows: {len(drop_index)}")

train.drop(drop_index,inplace=True)





plt.subplot(1,2,2)

ax = sns.boxplot(x="MSZoning", y="SalePrice", data=train);

ax = sns.swarmplot(x="MSZoning", y="SalePrice", data=train, color=".15")
# remove outliers with Density technique

def dbscan_outliers(var1,var2,dbscan_eps,dbscan_minsample,get_cluster_num):

    plt.figure(figsize=(8,8))

    scaler = RobustScaler()

    scale_var1 = scaler.fit_transform(pd.DataFrame((train[var1]))).reshape(1,-1)[0]

    scale_var2 = scaler.fit_transform(pd.DataFrame((train[var2]))).reshape(1,-1)[0]

    df_temp = pd.DataFrame({"var1":scale_var1 , "var2":scale_var2})

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_minsample).fit(df_temp.values) # parameters iteratively defined 

    df_temp["c"] = clustering.labels_

    df_temp.index = train.index

    sns.scatterplot(scale_var1,scale_var2, hue=clustering.labels_)

    return df_temp[df_temp["c"]==get_cluster_num].index

    
dropindex = dbscan_outliers("GrLivArea","SalePrice",dbscan_eps=.7,dbscan_minsample=10,get_cluster_num=-1)

print(f"Outliers droped {len(dropindex)}")

train.drop(dropindex,inplace=True)
# get quantitative and qualitative columns



quantitative_cols = [col for col in train.columns if train.dtypes[col] != 'object']



# remove SalePrice and Id from quantitative columns

quantitative_cols.remove('SalePrice')

quantitative_cols.remove('Id')

qualitative_cols = [col for col in train.columns if train.dtypes[col] == 'object']



print(f"Quantitative columns ({len(quantitative_cols)}): {quantitative_cols}", end="\n"*2)

print(f"Quanlitative columns ({len(qualitative_cols)}): {qualitative_cols}")

# concat train and test for preprocessing

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))





# get percent of missing values by columns

missing_serie = all_data.isna().sum().sort_values(ascending=False)



missing_serie[missing_serie>0].plot.bar(rot=90);

print("Missing percent")

print(missing_serie[missing_serie>0]/all_data.shape[0])
# case when null represent "No prescence" according to data description

all_data["PoolQC"].fillna("NoPool", inplace=True)

all_data["MiscFeature"].fillna("NOMiscFeature", inplace=True)

all_data["Alley"].fillna("NOAlley", inplace=True)

all_data["Fence"].fillna("NOFence", inplace=True)

all_data["FireplaceQu"].fillna("NOFireplaceQu", inplace=True)

all_data["GarageCond"].fillna("NOGarageCond", inplace=True) 

all_data["GarageQual"].fillna("NOGarageQual", inplace=True)

all_data["GarageYrBlt"].fillna("NOGarageYrBlt", inplace=True)

all_data["GarageFinish"].fillna("NOGarageFinish", inplace=True)

all_data["GarageType"].fillna("NOGarageType", inplace=True)

all_data["BsmtCond"].fillna("NOBsmtCond", inplace=True)

all_data["BsmtExposure"].fillna("NOBsmtExposure", inplace=True)

all_data["BsmtQual"].fillna("NOBsmtQual", inplace=True)

all_data["BsmtFinType2"].fillna("NOBsmtFinType2", inplace=True)

all_data["BsmtFinType1"].fillna("NOBsmtFinType1", inplace=True)





# Imputed with mean in numerical variables or mode (by neighborhood) in categorical

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

all_data["BsmtFinSF2"] = all_data.groupby("Neighborhood")["BsmtFinSF2"].transform(lambda x: x.fillna(x.median()))

all_data["BsmtFinSF1"] = all_data.groupby("Neighborhood")["BsmtFinSF1"].transform(lambda x: x.fillna(x.median()))

all_data["MasVnrType"] = all_data.groupby("Neighborhood")["MasVnrType"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["MasVnrArea"] = all_data.groupby("Neighborhood")["MasVnrArea"].transform(lambda x: x.fillna(x.mean()))

all_data["MSZoning"] = all_data.groupby("Neighborhood")["MSZoning"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["Exterior2nd"] = all_data.groupby("Neighborhood")["Exterior2nd"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["BsmtUnfSF"] = all_data.groupby("Neighborhood")["BsmtUnfSF"].transform(lambda x: x.fillna(x.median()))

all_data["TotalBsmtSF"] = all_data.groupby("Neighborhood")["TotalBsmtSF"].transform(lambda x: x.fillna(x.median()))

all_data["Exterior1st"] = all_data.groupby("Neighborhood")["Exterior1st"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["SaleType"] = all_data.groupby("Neighborhood")["SaleType"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["Electrical"] = all_data.groupby("Neighborhood")["Electrical"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["KitchenQual"] = all_data.groupby("Neighborhood")["KitchenQual"].transform(lambda x: x.fillna(x.mode().values[0]))

all_data["GarageArea"] = all_data.groupby("Neighborhood")["GarageArea"].transform(lambda x: x.fillna(x.median()))

all_data["GarageCars"] = all_data.groupby("Neighborhood")["GarageCars"].transform(lambda x: x.fillna(x.mode().values[0]))





# according to data description

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data["BsmtHalfBath"].fillna(0, inplace=True)

all_data["BsmtFullBath"].fillna(0, inplace=True)





all_data = all_data.drop(['Utilities'], axis=1) # no relevant according to value_counts

# apply logtransform to SalePrice

print("Skew before log transform:",train["SalePrice"].skew())

plt.subplot(1,2,1)

plt.title("SalePrice")

sns.distplot(train["SalePrice"])

plt.subplot(1,2,2)

plt.title("SalePrice Log")

sns.distplot(np.log1p(train["SalePrice"]));

train["SalePrice"] = np.log1p(train["SalePrice"])

print("Skew after log transform:",train["SalePrice"].skew())
# apply boxcox1p_transform in columns with abs(skew) >.7



skewed_cols = all_data[quantitative_cols].apply(lambda x: abs(skew(x.dropna())))

boxcox_transf_cols = skewed_cols[skewed_cols > 0.7].index



for col_box in boxcox_transf_cols:

    plt.subplot(1,2,1)

    plt.title(col_box)

    sns.distplot(all_data[col_box].dropna(), kde=False, axlabel=col_box)

    plt.subplot(1,2,2)

    plt.title(col_box + "_boxcox1p")

    sns.distplot(boxcox1p(all_data[col_box].dropna(), boxcox_normmax(all_data[col_box].dropna() + 1)), kde=False, axlabel= col_box+"_boxcox1p")

    all_data[col_box] = boxcox1p(all_data[col_box].dropna(), boxcox_normmax(all_data[col_box].dropna() + 1))

    plt.show()

    
# generate dummies

all_data = pd.get_dummies(all_data)
# regenerate train/test data

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold

# Setup cross validation folds

kf = KFold(n_splits=12, random_state=42, shuffle=True)



# ridge alphas

ridge_alphas = [1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]

param_grid = {'model__alpha':ridge_alphas}



# define pipeline model

ridge = Pipeline(steps=[('standardscaler', RobustScaler()),

                        ('model', Ridge(max_iter=1e7))])





search_ridge = GridSearchCV(ridge, param_grid,cv=kf,scoring="neg_mean_squared_error")

search_ridge.fit(X_train, y)



# get score of best estimator

np.sqrt(search_ridge.best_score_*-1)
cv_result = pd.DataFrame(search_ridge.cv_results_["params"])

cv_result["score"] = np.sqrt(search_ridge.cv_results_["mean_test_score"]*-1)

cv_result.plot(x="model__alpha",y="score");
alphas = np.array([ 0.01,0.005,0.001,0.0001,0.00001])



param_grid = {'model__alpha':alphas}



lasso = Pipeline(steps=[('standardscaler', RobustScaler()),

                    ('model', Lasso(max_iter=1e5))])





search_lasso = GridSearchCV(lasso, param_grid,cv=10,scoring="neg_mean_squared_error")

search_lasso.fit(X_train, y)

np.sqrt(search_lasso.best_score_*-1)
cv_result = pd.DataFrame(search_lasso.cv_results_["params"])

cv_result["score"] = np.sqrt(search_lasso.cv_results_["mean_test_score"]*-1)

cv_result.plot(x="model__alpha",y="score")
lasso_coef = pd.DataFrame(search_lasso.best_estimator_["model"].coef_,columns=["coef"])

lasso_coef["col"] = X_train.columns
coef_0 = (lasso_coef["coef"]==0).value_counts().get(True)

print(f"Lasso (coef==0) = {coef_0}")

lasso_coef[abs(lasso_coef["coef"])>0].sort_values(by="coef").plot(kind="bar",x="col",rot=80)
pred = search_lasso.best_estimator_.predict(X_test)

df_result = pd.DataFrame(np.expm1(pred),columns=["SalePrice"])

df_result["Id"] = test.Id

df_result.to_csv("submit.csv",index=False)
