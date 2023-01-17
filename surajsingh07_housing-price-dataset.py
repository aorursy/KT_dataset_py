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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')

sample_submission = pd.read_csv('/kaggle/input/home-data-for-ml-course/sample_submission.csv')
train_data.columns
features = ['LotArea','YearBuilt','YearRemodAdd','OverallCond','GarageArea','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr',

                 'TotRmsAbvGrd','SalePrice']

df_train = train_data[features]
df_train.head()
df_train.info()
df_train.describe()
df_train.hist(bins=30, figsize=(15,12), edgecolor='black')

plt.show()
sns.pairplot(df_train,corner=True)

plt.show()
plt.figure(figsize=(10,8))

matrix = np.triu(df_train.corr())

sns.heatmap(df_train.corr(), mask=matrix, annot=True)

plt.show()
# Finding out how much portion of data outliers takes

lotarea_outlier = (len(df_train.loc[df_train['LotArea']>40000])/len(df_train))*100

print('LotArea outlier percentage: {}%'.format(round(lotarea_outlier)))



# Removing the outliers

df_train = df_train.loc[df_train['LotArea']<40000]



df_train['LotArea'].hist(bins=30, edgecolor='black')

plt.title('LotArea Distribution')

plt.show()
# Making features from YearBulit and YearRemodAdd

newest_house_age = max(df_train['YearBuilt'])

df_train['HouseAge'] = df_train['YearBuilt'].apply(lambda year: newest_house_age - year)

df_train['LastMod'] = df_train['YearRemodAdd'].apply(lambda year: newest_house_age - year)
df_train.drop(['YearBuilt','YearRemodAdd'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_train.drop('SalePrice', axis=1), df_train['SalePrice'], test_size=0.3,

                                                   random_state=101)
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



param_grid = {'alpha':[0.1, 1, 10]}

grid = GridSearchCV(Ridge(), param_grid, refit=True, verbose=3)

grid.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error, mean_squared_error

pred = grid.predict(X_test)

print('mean absolute error: {}'.format(mean_absolute_error(y_test, pred)))

print('root mean squared error: {}'.format(np.sqrt(mean_squared_error(y_test,pred))))
print('performance on train set: {}'.format(grid.score(X_train,y_train)))

print('performance on testing set: {}'.format(grid.score(X_test,y_test)))
grid.best_params_
df_test = test_data[['LotArea','YearBuilt','YearRemodAdd','OverallCond','GarageArea','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr',

                 'TotRmsAbvGrd']]

df_test.head()
df_test = df_test.fillna(X_test['GarageArea'].mean())

df_test.isnull().sum()
# Making features from YearBulit and YearRemodAdd

newest_house_age = max(df_test['YearBuilt'])

df_test['HouseAge'] = df_test['YearBuilt'].apply(lambda year: newest_house_age - year)

df_test['LastMod'] = df_test['YearRemodAdd'].apply(lambda year: newest_house_age - year)



df_test.drop(['YearBuilt','YearRemodAdd'], axis=1, inplace=True)
pred = Ridge(alpha=10).fit(df_train.drop('SalePrice',axis=1),df_train['SalePrice']).predict(df_test)
output = pd.DataFrame({'Id':test_data.Id, 'SalePrice':pred})

output.to_csv('my_submission',index=False)

print('saved')