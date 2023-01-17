import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
# Import training/testing data

df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

df_train.head()


print (df_train.shape)

print (df_test.shape)
df_train_test = [df_train, df_test]
for df in df_train_test:

    print (df.get_dtype_counts())

    print (" ---------- ")
for df in df_train_test:

    total = df.isnull().sum().sort_values(ascending = False)

    total = total[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = total / len(df) * 100

    percent = percent[df.isnull().sum().sort_values(ascending = False) != 0]

    concat = pd.concat([total, percent], axis=1, keys=['Total','Percent'])

    print (concat)

    print ( "-------------")
# too many missing values for the following columns.. just drop them

for df in df_train_test:

    df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'],axis=1,inplace=True) 

    
df_train.drop(['Id'],axis=1,inplace=True)
df_test.drop(['Id'],axis=1,inplace=True)
df_train.groupby('Neighborhood')['LotFrontage'].mean()
df_train['LotFrontage'] = df_train.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))

df_test['LotFrontage'] = df_test.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
df_train.dropna(inplace = True)
print (df_train.isnull().any().sum())

print (df_test.isnull().any().sum())
print (df_train.shape)

print (df_test.shape)
sns.distplot(df_train["SalePrice"])
sns.boxplot(df_train["SalePrice"])
import scipy.stats as stats

stats.probplot(df_train["SalePrice"], plot = plt)
df_train = df_train[df_train["SalePrice"] < 700000]
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
stats.probplot(df_train["SalePrice"], plot = plt)
print(abs(df_train.corr())["SalePrice"].sort_values(ascending = False))
df_test.shape
y = df_train['SalePrice'].reset_index(drop=True)
y.head()
df_all = pd.concat((df_train, df_test)).reset_index(drop=True)

df_all.drop(["SalePrice"], axis = 1, inplace = True)

df_all.shape


df_all = pd.get_dummies(df_all, drop_first = True)

df_all.shape
print (df_all.get_dtype_counts())
for col in df_all.columns:

    print (col)
df_all['haspool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

df_all['hasgarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df_train_X = df_all.iloc[:len(y), :]

df_test_X = df_all.iloc[len(y):, :]
len(y) , df_test_X.shape, df_train_X.shape
from sklearn.model_selection import train_test_split

# Split up training set 

x_train, x_test, y_train, y_test = train_test_split(df_train_X, y, test_size=0.3)
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(n_estimators = 150 , max_features = 10 )
regr.fit(x_train, y_train)
print (regr.score (x_train, y_train))

print (regr.score(x_test, y_test))
predictions = regr.predict(x_test)
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test, predictions))
np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(predictions)))
feature_importances = pd.DataFrame(regr.feature_importances_, index = df_all.columns, 

                                   columns=['importance']).sort_values('importance', ascending=False)

feature_importances.head(20)
plt.scatter(np.expm1(y_test),np.expm1(predictions))

plt.title('Predicted vs. Actual')

plt.xlabel('Actual Sale Price')

plt.ylabel('Predicted Sale Price')
df_test_X.head()
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.head()
df_test_X.shape
df_test_X = df_test_X.fillna(0)

pred = np.expm1(regr.predict(df_test_X))

pred.shape
submission["SalePrice"] = np.expm1(regr.predict(df_test_X))

submission.head()
submission.to_csv("submission.csv", index=False)