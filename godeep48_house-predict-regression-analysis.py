import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder



from prettytable import PrettyTable



from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, mean_absolute_error, mean_squared_log_error

from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score



from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df.head(5)
#COlumns available

df.columns
#Shape of DataFrame

print(f"Shape of Dataframe: {df.shape}")
#The target Variable

df["SalePrice"].describe()
#Distribution plot of Traget Variable

sns.set_style(style="whitegrid")

sns.distplot(df["SalePrice"])

plt.title("Distribution plot")

plt.show()
print(f"The skewness is: {df['SalePrice'].skew()}")

print(f"The kurtosis is: {df['SalePrice'].kurt()}")
#Variable which have data types int64 and float64

print(df.dtypes[df.dtypes!=object])

print(f"Total Numbers of Numerical Variables: {len(df.dtypes[df.dtypes!=object])}")
#Variable which have data types int64 and float64

var_num = df.dtypes[df.dtypes!=object].index.values.tolist()
##df_num = df[var_num]

df_num = df[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]
#Dataframe with numerical values

df_num.head(5)
len(df)
##Seperating variable w.r.t the contunuous and discrete values

var_list_continuous = []

var_list_discrete = []

for i in df_num.columns:

    count = len(df_num[i].value_counts())

    if count>=15:

        var_list_continuous.append(i)

    else:

        var_list_discrete.append(i)
print(f"Continuous Features: {var_list_continuous}")

print(f"Discrete Features: {var_list_discrete}")
print(f"Length of variable having continuous values: {len(var_list_continuous)}")
# Create new dataframe with continuous values.

df_continuous = df_num[var_list_continuous]

df_continuous.head(5)
## Correlation Coeficiant 

#Heatmap of corelation of Numerical Variables.

plt.figure(figsize=(15,15))

sns.heatmap(df_continuous.corr(), cmap="Blues", annot=True)

plt.show()
#HIghest correlated variables with SalePrice ie greater than 0.4

correlation_threshold = 0.4



df_continuous.corr()["SalePrice"][~df_continuous.corr()["SalePrice"].between(-correlation_threshold, correlation_threshold)]
imp_var = list(df_continuous.corr()["SalePrice"][~df_continuous.corr()["SalePrice"].between(-correlation_threshold, correlation_threshold)].keys())

imp_var.remove("SalePrice")
plt.figure(1)

fig_no = 1

n_row = len(imp_var)/3 if len(imp_var)%3==0 else int(len(imp_var)/3)+1

plt.figure(figsize=(20,20))

for i in imp_var:

    plt.subplot(n_row,3,fig_no)

    fig_no+=1

    sns.scatterplot(df_num[i], df_num["SalePrice"])

plt.show()
len(var_list_continuous)
#To find the outliers.



plt.figure(1)

fig_no = 1

plt.figure(figsize=(20,15))

for i in var_list_continuous:

    plt.subplot(6,4,fig_no)

    fig_no+=1

    sns.boxplot(y=df_num[i])

plt.show()
outliers_var = ['LotArea', 'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']

df_outlier = df_num[outliers_var]
from scipy import stats

import numpy as np

z = np.abs(stats.zscore(df_outlier))

print(z)
## Filter the value which is greater than 3. 

threshold=3.0

df_outliers = pd.DataFrame(z, columns=outliers_var)>threshold



## Find the number of index where we have atleast a single outlier value in any feature.

list_outliers = []

for i in range(0, len(df_outliers)):

    for val in df_outliers.loc[i]:

        if val:

            list_outliers.append(i)
print(f"Total number of rows which having atleast a single outlier value: {len(set(list_outliers))}")

print(f"Percent of total rows we need to drop due to outliers: {np.round(len(set(list_outliers))/len(df_num)*100, 2)}%")
df_discrete = df_num[var_list_discrete]

df_discrete["SalePrice"] = df_num["SalePrice"]

df_discrete.head(5)
print(f"Number of features with discrete values: {len(var_list_discrete)}")
plt.figure(figsize=(15,15))

plt.title("Heatmap of Correlation Coefficient of each variables")

sns.heatmap(df_discrete.corr(), cmap="Blues", annot=True)

plt.show()
#Correlation which is greater than threshold.

correlation_threshold = 0.4

df_discrete.corr()["SalePrice"][~df_discrete.corr()["SalePrice"].between(-correlation_threshold, correlation_threshold)]
imp_var = list(df_discrete.corr()["SalePrice"][~df_discrete.corr()["SalePrice"].between(-correlation_threshold, correlation_threshold)].keys())

imp_var.remove("SalePrice")
plt.figure(1)

fig_no = 1

n_row = len(imp_var)/3 if len(imp_var)%3==0 else int(len(imp_var)/3)+1

plt.figure(figsize=(20,20))

for i in imp_var:

    plt.subplot(n_row,3,fig_no)

    fig_no+=1

    sns.boxplot(df_num[i], df_num["SalePrice"])

plt.show()
#Variable which have data types object

var_cat = df.dtypes[df.dtypes==object].index.values.tolist()

#Number of variable which have dtype object.

print(f"Number of variables with datatypes object: {len(var_cat)}")
#Dataframe with object datatype variables

df_cat = df[var_cat]

df_cat.head(5)
plt.figure(1)

fig_no = 1

plt.figure(figsize=(20,20))

for i in df_cat.columns.values.tolist():

    plt.subplot(11,4,fig_no)

    fig_no+=1

    sns.boxplot(df_cat[i], df["SalePrice"])

plt.show()
#Dealing with NULL values

df_null = pd.DataFrame(df.isnull().sum()[df.isnull().sum()>0], columns=["Sum of Null Values"])

df_null["Percent of NULL (%)"] = df.isnull().mean().round(4)[df.isnull().mean()>0]*100

df_null
#Removing variables having more than 40% of NULL values.

df = df[(df.isnull().mean()[df.isnull().mean()<0.4]).keys().tolist()]
df_num = df[['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold', 'SalePrice']]
#Dealing with NULL values

df_null = pd.DataFrame(df_num.isnull().sum()[df_num.isnull().sum()>0], columns=["Sum of Null Values"])

df_null["Percent of NULL (%)"] = df_num.isnull().mean().round(4)[df_num.isnull().mean()>0]*100

df_null
#Since there are possible outliers, we can impute missing value with median rather than mean.



imp = SimpleImputer(strategy="median")

df_num = pd.DataFrame(imp.fit_transform(df_num), columns=df_num.columns, index=df_num.index)



#Display the NULL values

df_null_num = pd.DataFrame(df_num.isnull().sum()[df_num.isnull().sum()>0], columns=["NULL sum"])

df_null_num["Percent of NULL (%)"] = (df_num.isnull().mean()[df_num.isnull().mean()>0])*100

df_null_num
#Variable which have data types object

var_cat = df.dtypes[df.dtypes==object].index.values.tolist()

#Dataframe with object datatype variables

df_cat = df[var_cat]

df_cat.head(5)
#Dealing with NULL values

df_null = pd.DataFrame(df_cat.isnull().sum()[df_cat.isnull().sum()>0], columns=["Sum of Null Values"])

df_null["Percent of NULL (%)"] = df_cat.isnull().mean().round(4)[df_cat.isnull().mean()>0]*100

df_null
#Replace the NULL value with the most frequent values.

imp = SimpleImputer(strategy="most_frequent")

df_cat = pd.DataFrame(imp.fit_transform(df_cat), columns=df_cat.columns, index=df_cat.index)



#Display the NULL values.

perc_Df = pd.DataFrame(df_cat.isnull().sum()[df_cat.isnull().sum()>0], columns=["NULL sum"])

perc_Df["NULL percent"] = (df_cat.isnull().mean()[df_cat.isnull().mean()>0])*100

perc_Df
encode = OneHotEncoder(drop='first',sparse=False)

encode.fit(df_cat)



df_cat_dummies = encode.transform(df_cat)

df_cat_dummies = pd.DataFrame(df_cat_dummies, columns=encode.get_feature_names(), index=df_cat.index)

df_cat_dummies.head(5)
print(f"The shape of Numerical variable: {df_num.shape}")

print(f"The shape of categorical variable: {df_cat.shape}")

print(f"The shape of categoricall dummy variables: {df_cat_dummies.shape}")
df_cat_dummies["SalePrice"] = df_num["SalePrice"]



df_cat_dummies.corr()["SalePrice"][df_cat_dummies.corr()["SalePrice"]>=correlation_threshold]
print(list(df_cat_dummies.corr()["SalePrice"][df_cat_dummies.corr()["SalePrice"]>=correlation_threshold].keys()))
df_all_var = df_num.join(df_cat_dummies.drop("SalePrice", axis=1))



print(f"Shape of Final ALL variable Dataframe: {df_all_var.shape}")
X = df_all_var.drop('SalePrice', axis=1)

y = df_all_var["SalePrice"]



print(f"X Train dataset shape: {X.shape}")

print(f"Y train dataset shape: {y.shape}")



r2 = make_scorer(r2_score, greater_is_better=True)

rmse = make_scorer(mean_squared_error,greater_is_better=False,squared=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")

print(f"y_train shape: {y_train.shape}")

print(f"y_test shape: {y_test.shape}")
model_name = "LinearRegression"

model=LinearRegression()



param_grid = [{model_name+'__fit_intercept':[True,False]}]

pipeline = Pipeline([(model_name, model)])

regressor = GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
regressor.fit(X_train, y_train)

print(f"Best score: {regressor.best_score_}")

print(f"Best params: {regressor.best_params_}")
y_pred = regressor.predict(X_test)



mae_linear = mean_absolute_error(y_test, y_pred)

print("Model 1 MAE: %d" % (mae_linear))



r2_val_linear = r2_score(y_test, y_pred)

print(f"R2 Score: {r2_val_linear}")
pd.DataFrame({"y_test":y_test, "y_pred":y_pred}).head(5)
model_name = "Lasso"

model=Lasso()



alpha = [2**i for i in range(-5, 15)]

param_grid = [  {model_name+'__'+'alpha': alpha}]
pipeline = Pipeline([(model_name, model)])



reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X_train,y_train.to_numpy())
print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)
y_pred = reg.predict(X_test)

mae_lasso, rmsle_lasso = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_log_error(y_test, y_pred))

print("Model 1 MAE: %d, RMSLE: %f" % (mae_lasso, rmsle_lasso))



r2_value_lasso = r2_score(y_test, y_pred)

print(f"R2 Score: {r2_value_lasso}")
pd.DataFrame({"y_test":y_test, "y_pred":y_pred}).head(5)
model_name = "Ridge"

model=Ridge()



alpha = [2**i for i in range(-5, 15)]

param_grid = [{model_name+'__'+'alpha': alpha}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X_train,y_train.to_numpy())
print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)
y_pred = reg.predict(X_test)



mae_ridge, rmsle_ridge = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_log_error(y_test, y_pred))

print("Model 1 MAE: %d, RMSLE: %f" % (mae_ridge, rmsle_ridge))



r2_value_ridge = r2_score(y_test, y_pred)

print(f"R2 Score: {r2_value_ridge}")
pd.DataFrame({"y_test":y_test, "y_pred":y_pred}).head(5)
table = PrettyTable()

table.field_names = ["Model", "MAE", "RMSLE", "R-Squared Value"]

table.add_row(["Linear Regression", mae_linear, "NA", r2_val_linear])

table.add_row(["LASSO Regression", mae_lasso, rmsle_lasso, r2_value_lasso])

table.add_row(["RIDGE Regression", mae_ridge, rmsle_ridge, r2_value_ridge])



print(table)
imp_var_dis = list(df_discrete.columns)



#Variable with higher correlation coefficiant

imp = ['OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'SalePrice' ]



#Removing variable having higher correlation coefficiant

for i in imp:

    imp_var_dis.remove(i)
new_df_discrete = df_discrete[imp_var_dis]

new_df_discrete.head(5)
## Changing the numerical values of discrete variables into string. So that the columns would never 

## repeat while One hot encoding.

for col in new_df_discrete.columns:

    for i in range(len(new_df_discrete[col])):

        new_df_discrete[col][i] = col+str(new_df_discrete[col][i])

        #new_df_discrete[col][]
new_df_discrete.head(5)
## Introducing OneHotEncoding

encode = OneHotEncoder(drop='first',sparse=False)

encode.fit(new_df_discrete)



df_discrete_dummies = encode.transform(new_df_discrete)

df_discrete_dummies = pd.DataFrame(df_discrete_dummies, columns=encode.get_feature_names(), index=new_df_discrete.index)

df_discrete_dummies.head(5)
## Join the one hot encoded variable with the variables having highest correlation.

df_final_discrete = df_discrete[imp].join(df_discrete_dummies)

df_final_cont = df_num[list(df_continuous.columns)]



# Drop Target Variable from dataframe

df_final_discrete = df_final_discrete.drop("SalePrice", axis=1)

print(df_final_discrete.shape)

# Drop Target Variable from dataframe

df_final_cont = df_final_cont.drop("SalePrice", axis=1)

print(df_final_cont.shape)

print(df_cat_dummies.shape)
# Concatenate all the dataframes to form final DF for modeling 

df_final = df_final_discrete.join(df_final_cont.join(df_cat_dummies))
## Normalise final dataframe using RobustScaler.



def normalise_encode(dataframe):

    columns = list(dataframe.columns)

    target_var = dataframe["SalePrice"]

    dataframe = dataframe.drop("SalePrice", axis=1)



    scaler = RobustScaler()

    dataframe = scaler.fit_transform(dataframe)

    dataframe = pd.DataFrame(dataframe, columns=columns.remove("SalePrice"))

    dataframe["SalePrice"] = target_var

    

    return dataframe



#df_final = normalise_encode(df_final)
X = df_final.drop("SalePrice", axis=1)

y = df_final["SalePrice"]

print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print(f"X_train shape: {X_train.shape}")

print(f"X_test shape: {X_test.shape}")

print(f"y_train shape: {y_train.shape}")

print(f"y_test shape: {y_test.shape}")
r2 = make_scorer(r2_score, greater_is_better=True)

rmse = make_scorer(mean_squared_error,greater_is_better=False,squared=False)
model_name = "LinearRegression"

model=LinearRegression()



param_grid = [{model_name+'__fit_intercept':[True,False]}]

pipeline = Pipeline([(model_name, model)])

regressor = GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)
regressor.fit(X_train, y_train)

print(f"Best score: {regressor.best_score_}")

print(f"Best params: {regressor.best_params_}")
y_pred = regressor.predict(X_test)



mae_linear_enc = mean_absolute_error(y_test, y_pred)

print("Model 1 MAE: %d" % (mae_linear_enc))



r2_val_linear_enc = r2_score(y_test, y_pred)

print(f"R2 Score: {r2_val_linear_enc}")
pd.DataFrame({"y_test":y_test, "y_pred":y_pred}).head(5)
model_name = "Lasso"

model=Lasso()



alpha = [2**i for i in range(-5, 15)]

param_grid = [  {model_name+'__'+'alpha': alpha}]
pipeline = Pipeline([(model_name, model)])



reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X_train,y_train.to_numpy())
print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)
y_pred = reg.predict(X_test)

mae_lasso_enc, rmsle_lasso_enc = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_log_error(y_test, y_pred))

print("Model 1 MAE: %d, RMSLE: %f" % (mae_lasso_enc, rmsle_lasso_enc))



r2_value_lasso_enc = r2_score(y_test, y_pred)

print(f"R2 Score: {r2_value_lasso_enc}")
pd.DataFrame({"y_test":y_test, "y_pred":y_pred}).head(5)
model_name = "Ridge"

model=Ridge()



alpha = [2**i for i in range(-5, 15)]



param_grid = [{model_name+'__'+'alpha': alpha}]
pipeline = Pipeline([(model_name, model)])





reg=GridSearchCV(pipeline,param_grid,cv=5, scoring=rmse, n_jobs=-1)

reg.fit(X_train,y_train.to_numpy())
print('best training param:',reg.best_params_)

print('best training score rmse', reg.best_score_)
y_pred = reg.predict(X_test)



mae_ridge_enc, rmsle_ridge_enc = mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_log_error(y_test, y_pred))

print("Model 1 MAE: %d, RMSLE: %f" % (mae_ridge_enc, rmsle_ridge_enc))



r2_value_ridge_enc = r2_score(y_test, y_pred)

print(f"R2 Score: {r2_value_ridge_enc}")
pd.DataFrame({"y_test":y_test, "y_pred":y_pred}).head(5)
table = PrettyTable()

table.field_names = ["Model", "MAE", "RMSLE", "R-Squared Value"]

table.add_row(["Linear Regression", mae_linear_enc, "NA", r2_val_linear_enc])

table.add_row(["LASSO Regression", mae_lasso_enc, rmsle_lasso_enc, r2_value_lasso_enc])

table.add_row(["RIDGE Regression", mae_ridge_enc, rmsle_ridge_enc, r2_value_ridge_enc])



print(table)