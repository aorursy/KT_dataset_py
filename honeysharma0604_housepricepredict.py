# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Import libraries

import pandas_profiling as pp

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
## input data

X_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

X_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
## combine data for imputation 



X_test['train'] = 0

X_train['train'] = 1



all_data = pd.concat([X_train,X_test])

X_train.head(5)
X_train.info()
X_train.describe() ## this only include numeric data
X_test.info()
X_test.head(5)
X_test.describe() 
for i in X_train.columns:

    print("Basic Stats for feature: {0}".format(i))

    print(X_train[i].describe())

    print("=====================================")
for i in X_test.columns:

    print("Basic stats for feature: {0}".format(i))

    print(X_test[i].describe())

    print("===========================")
X_train.isna().sum()
X_test.isna().sum()
## insights of train data

profile=pp.ProfileReport(X_train, minimal = True)

profile.to_file("output.html")
## insights of train data

profile=pp.ProfileReport(X_test)

profile.to_file("Testoutput.html")
X_train.SalePrice.describe()
plt.hist([X_train.SalePrice])
print ("Skew of SalePrice:", X_train['SalePrice'].skew())
X_train['LT_SalePrice'] = np.log(X_train['SalePrice']+1)



plt.hist(X_train.LT_SalePrice, color = 'Green')
print("skew is LT_SalePrice: ", (X_train.LT_SalePrice.skew()))
## Method to draw corr heatmap



def correlation_heatmap(data):

    corr = data.corr()

    mask = np.zeros_like(corr)

    mask[np.triu_indices_from(mask)] = True

    fig,ax =plt.subplots(figsize =(25,25))

    sns.heatmap(data=corr, vmax=1.0, center=0, fmt='.2f',

                square=True, linewidths=.5,  cbar_kws={"shrink": .70}, annot=True,mask=mask)
## Visualize the train data corr:



correlation_heatmap(X_train)
## Visualize the test data corr:



correlation_heatmap(X_test)
## Only Keeping Colummns with Corr > .50



X_test_corr = X_test[['Id','OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea']]

#y = X_train['SalePrice']

X_train_corr = X_train[['Id','OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'SalePrice']]



# X_test_corr['train'] = 0

# X_train_corr['train'] = 1

# all_data = pd.concat([X_train_corr,X_test_corr])

# all_data.reset_index(inplace = True)
# ## Removing NULL values in Train Data



# null_values = pd.DataFrame(X_train.isnull().sum().sort_values(ascending = False)[:16])

# null_values.index.name = 'Feature'

# null_values.columns = ['Number of NULL values']

# null_values
# null_rows = X_train.iloc[np.where(X_train.isnull())]

# print(null_rows)
# null_rows2 = X_test.iloc[np.where(X_test.isnull())]

# print(null_rows2)
null_values1 = pd.DataFrame(X_train_corr.isnull().sum().sort_values(ascending=False)[:5])

null_values2 = pd.DataFrame(X_test_corr.isnull().sum().sort_values(ascending=False)[:5])
#all_data.info() ## no categorical values
## impute null values

## GarageYrBlt

X_train_corr['GarageYrBlt'].fillna(X_train_corr['YearBuilt'], inplace = True)

X_test_corr['GarageYrBlt'].fillna(X_test_corr['YearBuilt'], inplace = True)



## other columns

X_train_corr = X_train_corr.interpolate().dropna()

X_test_corr = X_test_corr.interpolate().dropna()
nan_rows = all_data.iloc[np.where(all_data.isna())]

nan_rows
## Deal with Outliers

data = X_train_corr



from sklearn.ensemble import IsolationForest as IF

clf = IF(max_samples = 100, random_state = 1)

clf.fit(data)

y_noano = clf.predict(data)

y_noano = pd.DataFrame(y_noano, columns = ['Top'])



data = data.iloc[y_noano[y_noano['Top'] == 1].index.values]

data.reset_index(drop = True, inplace = True)

print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])

print("Number of rows without outliers:", data.shape[0])





y = data.SalePrice



data.drop(columns=['SalePrice'],inplace=True)



print(data.info())
## Spilt the test and train data 



X_train_dummies = data

X_test_dummies = X_test_corr

X_test_dummies.shape, X_train_dummies.shape
# Break off validation set from training data



from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X_train_dummies, y,train_size=0.8, test_size=0.2,random_state=0)
## Model XGBOOST

from sklearn.metrics import mean_absolute_error



from sklearn.model_selection import cross_val_score



from xgboost import XGBRegressor



xgb = XGBRegressor(random_state =1,n_estimators=10000, learning_rate=0.01, n_jobs=4)

cv = cross_val_score(xgb,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
xgb.fit(X_train,y_train,early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)
predictions_1=xgb.predict(X_valid)

mae_1 = mean_absolute_error(predictions_1,y_valid)

print(mae_1)
predictions_2 = xgb.predict(X_test_dummies)

predictions_2=predictions_2.astype(float)

predictions_2
# Save test predictions to file



output = pd.DataFrame({'Id': X_test.Id,'SalePrice': predictions_2})

print(output)

output.to_csv('submission1.csv', index=False)