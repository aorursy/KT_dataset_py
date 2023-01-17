import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

%matplotlib inline

from sklearn.model_selection import cross_val_score, train_test_split, KFold, cross_val_predict

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, BayesianRidge, LassoLarsIC

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import math
data_train = pd.read_csv("/kaggle/input/train.csv")

data_train
data_test = pd.read_csv("/kaggle/input/test.csv")

data_test
y = data_train.SalePrice

train_without_response = data_train[data_train.columns.difference(['SalePrice'])]

result = pd.concat([train_without_response,data_test], ignore_index=True)

result
result.head()
result.tail()
result.info()
result.shape #Numero di colonne e righe 
result.columns
result.describe()
y.describe()
sns.distplot(data_train['SalePrice']);

print("Skewness coeff. is: %f" % data_train['SalePrice'].skew())

print("Kurtosis coeff. is: %f" % data_train['SalePrice'].kurt())
data_year_trend = pd.concat([data_train['SalePrice'], data_train['YearBuilt']], axis=1)

data_year_trend.plot.scatter(x='YearBuilt', y='SalePrice', ylim=(0,800000));
data_bsmt_trend = pd.concat([data_train['SalePrice'], data_train['TotalBsmtSF']], axis=1)

data_bsmt_trend.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
data_GrLivArea_trend = pd.concat([data_train['SalePrice'], data_train['GrLivArea']], axis=1)

data_GrLivArea_trend.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
data_PoolArea_trend = pd.concat([data_train['SalePrice'], data_train['PoolArea']], axis=1)

data_PoolArea_trend.plot.scatter(x='PoolArea', y='SalePrice', ylim=(0,800000));
data = pd.concat([data_train['SalePrice'], data_train['OverallQual']], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
corr_matrix = result.corr()

f, ax1 = plt.subplots(figsize=(12,9)) 

ax1=sns.heatmap(corr_matrix,vmax = 0.9); 
corrmat = data_train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(9,9))

g = sns.heatmap(data_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
var = data_train[data_train.columns[1:]].corr()['SalePrice'][:]

var
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(data_train[cols], height = 2.5)

plt.show();
total_null = result.isnull().sum().sort_values(ascending=False) #First sum and order all null values for each variable

percentage = (result.isnull().sum()/result.isnull().count()).sort_values(ascending=False) #Get the percentage

missing_data = pd.concat([total_null, percentage], axis=1, keys=['Total', 'Percentage'])

missing_data.head(20)
result = result.drop((missing_data[missing_data["Percentage"] > 0.15]).index,1) #Drop All Var. with null values > 1

#data_train = data_train.drop(data_train.loc[data_train['Electrical'].isnull()].index) #Delete the single null value in Electrical

result.isnull().sum()
del result["KitchenAbvGr"]

del result["YrSold"]

del result["MoSold"]

del result["MiscVal"]

del result["ScreenPorch"]

del result["X3SsnPorch"]

del result["BsmtHalfBath"]

del result["LowQualFinSF"]

del result["OverallCond"]

del result["EnclosedPorch"]

del result["MSSubClass"]

del result["X1stFlrSF"]

del result["YearBuilt"]

del result["YearRemodAdd"] 

del result["BsmtFinSF2"] #0 variance

del result["BsmtFinSF1"] #Because BsmtFinSF1 + BsmtUnfSF + BsmtFinSF2 = TotalBsmtSF

del result["BsmtUnfSF"] #Because BsmtFinSF1 + BsmtUnfSF + BsmtFinSF2 = TotalBsmtSF

del result["PoolArea"] #0 variance

del result["GarageYrBlt"] #Dropped for the same reason of YearBuilt, it might mislead our predictions

del result["GarageCond"] #0 Variance

del result["GarageArea"] #High Correlation

del result["TotRmsAbvGrd"] #High Correlation

result
result['ExterCond'].value_counts()
del result["Street"]

del result["LandContour"]

del result["Utilities"]

del result["LandSlope"]

del result["Condition2"]

del result["RoofMatl"]

del result["BsmtFinType2"] #0 variance

del result["Electrical"] #0 Variance

del result["Condition1"]#Too many levels versione 2

del result["BldgType"]#versione 2

del result["HouseStyle"]#versione 2

del result["Exterior1st"]#versione 2

del result["Exterior2nd"]#versione 2

del result["Foundation"]#versione 2

del result["CentralAir"]#0 variance

del result["Functional"]#0 variance

del result["SaleType"]#0 variance

del result["SaleCondition"]#0 variance

del result["RoofStyle"]#0 variance

result
result.shape
#Here we encode ExterQual in a rank

result.loc[result['ExterQual'] == "Ex", 'ExterQual'] = 5

result.loc[result['ExterQual'] == "Gd", 'ExterQual'] = 4

result.loc[result['ExterQual'] == "TA", 'ExterQual'] = 3

result.loc[result['ExterQual'] == "Fa", 'ExterQual'] = 2

result.loc[result['ExterQual'] == "Po", 'ExterQual'] = 1

result['ExterQual']
#Here we encode ExterCond in Rank

result.loc[result['ExterCond'] == "Ex", 'ExterCond'] = 5

result.loc[result['ExterCond'] == "Gd", 'ExterCond'] = 4

result.loc[result['ExterCond'] == "TA", 'ExterCond'] = 3

result.loc[result['ExterCond'] == "Fa", 'ExterCond'] = 2

result.loc[result['ExterCond'] == "Po", 'ExterCond'] = 1

result['ExterCond']
#Here we encode HeatingQC in Rank

result.loc[result['HeatingQC'] == "Ex", 'HeatingQC'] = 5

result.loc[result['HeatingQC'] == "Gd", 'HeatingQC'] = 4

result.loc[result['HeatingQC'] == "TA", 'HeatingQC'] = 3

result.loc[result['HeatingQC'] == "Fa", 'HeatingQC'] = 2

result.loc[result['HeatingQC'] == "Po", 'HeatingQC'] = 1

result['HeatingQC']
#Here we encode BsmtFinType1 in Rank

result.loc[result['BsmtFinType1'] == "GLQ", 'BsmtFinType1'] = 6

result.loc[result['BsmtFinType1'] == "ALQ", 'BsmtFinType1'] = 5

result.loc[result['BsmtFinType1'] == "BLQ", 'BsmtFinType1'] = 4

result.loc[result['BsmtFinType1'] == "Rec", 'BsmtFinType1'] = 3

result.loc[result['BsmtFinType1'] == "LwQ", 'BsmtFinType1'] = 2

result.loc[result['BsmtFinType1'] == "Unf", 'BsmtFinType1'] = 1

result['BsmtFinType1'].fillna(0, inplace= True)

result['BsmtFinType1']
#Here we encode BsmtCond in Rank

result.loc[result['BsmtCond'] == "Ex", 'BsmtCond'] = 5

result.loc[result['BsmtCond'] == "Gd", 'BsmtCond'] = 4

result.loc[result['BsmtCond'] == "TA", 'BsmtCond'] = 3

result.loc[result['BsmtCond'] == "Fa", 'BsmtCond'] = 2

result.loc[result['BsmtCond'] == "Po", 'BsmtCond'] = 1

result['BsmtCond'].fillna(0, inplace= True)

result['BsmtCond']
#Here we encode BsmtQual in Rank

result.loc[result['BsmtQual'] == "Ex", 'BsmtQual'] = 5

result.loc[result['BsmtQual'] == "Gd", 'BsmtQual'] = 4

result.loc[result['BsmtQual'] == "TA", 'BsmtQual'] = 3

result.loc[result['BsmtQual'] == "Fa", 'BsmtQual'] = 2

result.loc[result['BsmtQual'] == "Po", 'BsmtQual'] = 1

result['BsmtQual'].fillna(0, inplace= True)

result['BsmtQual']
#Here we encode KitchenQual in Rank

result.loc[result['KitchenQual'] == "Ex", 'KitchenQual'] = 4

result.loc[result['KitchenQual'] == "Gd", 'KitchenQual'] = 3

result.loc[result['KitchenQual'] == "TA", 'KitchenQual'] = 2

result.loc[result['KitchenQual'] == "Fa", 'KitchenQual'] = 1

result['KitchenQual']
#Here we encode BsmtExposure in Rank

result.loc[result['BsmtExposure'] == "Gd", 'BsmtExposure'] = 4

result.loc[result['BsmtExposure'] == "Av", 'BsmtExposure'] = 3

result.loc[result['BsmtExposure'] == "Mn", 'BsmtExposure'] = 2

result.loc[result['BsmtExposure'] == "No", 'BsmtExposure'] = 1

result['BsmtExposure'].fillna(0, inplace= True)

result['BsmtExposure']
#Here we encode GarageQual in Rank

result.loc[result['GarageQual'] == "Ex", 'GarageQual'] = 5

result.loc[result['GarageQual'] == "Gd", 'GarageQual'] = 4

result.loc[result['GarageQual'] == "TA", 'GarageQual'] = 3

result.loc[result['GarageQual'] == "Fa", 'GarageQual'] = 2

result.loc[result['GarageQual'] == "Po", 'GarageQual'] = 1

result['GarageQual'].fillna(0, inplace= True)

result['GarageQual']
del result["GarageQual"]#perchè tutti i valori sono 3
#Here we encode GarageFinish in Rank

result.loc[result['GarageFinish'] == "Fin", 'GarageFinish'] = 4

result.loc[result['GarageFinish'] == "RFn", 'GarageFinish'] = 3

result.loc[result['GarageFinish'] == "Unf", 'GarageFinish'] = 2

result['GarageFinish'].fillna(0, inplace= True)

result['GarageFinish']
#HERE WE FILL THE LAST NAs IN THOSE VARIABLES WHICH WE CAN NOT RANK

result['MasVnrType'].fillna("None", inplace= True)
result['MasVnrArea'].fillna(0, inplace= True)
result['GarageType'].fillna("No Garage", inplace= True)
#Correlation matrix with new encoded variables

corr_matrix = result.corr()

f, ax1 = plt.subplots(figsize=(25,25)) #Crea il sistema di riferimento

ax1=sns.heatmap(corr_matrix,vmax = 0.9); #Con Seaborn fai una heatmap che ha val. max. 0.9
corrmat = data_train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.3]

plt.figure(figsize=(9,9))

g = sns.heatmap(data_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
pd.set_option('display.max_columns', 70)
#Near 0 variance

del result['ExterCond']
del result['BsmtCond'] #Near 0 variance
#Here we extract the numerical variables, this will come in handy later on

n_features = result.select_dtypes(exclude = ["object"]).columns
def mod_outlier(df):

        df1 = df.copy()

        df = df._get_numeric_data()





        q1 = df.quantile(0.25)

        q3 = df.quantile(0.75)



        iqr = q3 - q1



        lower_bound = q1 -(1.5 * iqr) 

        upper_bound = q3 +(1.5 * iqr)





        for col in df.columns:

            for i in range(0,len(df[col])):

                if df[col][i] < lower_bound[col]:            

                    df[col][i] = lower_bound[col]



                if df[col][i] > upper_bound[col]:            

                    df[col][i] = upper_bound[col]    





        for col in df.columns:

            df1[col] = df[col]



        return(df1)



result = mod_outlier(result)
for i in result[n_features]:

    sns.boxplot(x=result[i])

    plt.show()
result
#Here we split train and test back and we attach "SalePrice" to the train

data_train_new, data_test_new = result[:1100], result[1101:]

data_train_new['SalePrice'] = y
data_train_new
data_test_new
data_train_dummies = pd.get_dummies(data_train_new)

data_train_dummies
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X.values)

    rmse= np.sqrt(-cross_val_score(model, X.values, Y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
X = data_train_dummies[data_train_dummies.columns.difference(['SalePrice'])]
Y = data_train_dummies['SalePrice']
X
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .30, random_state = 40)
lr = LinearRegression()

lr.fit(X_train,Y_train)
print(lr.intercept_)
print(lr.coef_)
predicted = lr.predict(X_test)

plt.figure(figsize=(15,8))

plt.scatter(Y_test,predicted)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')

plt.show()
score = rmsle_cv(lr)

print("\nLinear Regression score: {:.4f}\n".format(score.mean()))
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(Y_test, predicted))

print('MSE:', metrics.mean_squared_error(Y_test, predicted))

print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predicted)))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
##lasso = linear_model.Lasso()

### y_pred = cross_val_predict(lasso, X, y, cv=5)
GBoost = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

#RMSE estimated through the partition of the train set

GBoost.fit(X_train, Y_train)

rmse = math.sqrt(mean_squared_error(Y_test, GBoost.predict(X_test)))

print("RMSE: %.4f" % rmse)
score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
regressor = RandomForestRegressor(n_estimators=300, random_state=0)

regressor.fit(X,Y)



# Score model

score = rmsle_cv(regressor)

print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))