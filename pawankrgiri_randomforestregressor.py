# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')

train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
train.head()
train= train.dropna(axis = 1)

train.head()

train.columns
corr = train.corr()

plt.figure(figsize=(50,50))

sns.set(font_scale=2)

sns.heatmap(corr,vmax=0.8,annot=True,square=True)

plt.show()
features=['OverallQual','YearBuilt','TotalBsmtSF','GrLivArea','FullBath','GarageArea']

train = train.dropna(axis=0)

x_train=train[features]

y_train=train['SalePrice']

x_train
sns.set()

sns.scatterplot(x=x_train['OverallQual'],y=train['SalePrice'])
sns.scatterplot(x=y_train,y=x_train['YearBuilt'])
sns.scatterplot(x=x_train['FullBath'],y=y_train)
sns.scatterplot(x=x_train['GarageArea'],y=y_train)
sns.scatterplot(x=x_train['GrLivArea'],y=y_train)
sns.scatterplot(x=x_train['TotalBsmtSF'],y=y_train)
x_train.sort_values(by = 'TotalBsmtSF', ascending = False)[:1]
x_train.drop(x_train[x_train['TotalBsmtSF']>5800].index,inplace=True)

x_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

x_train.drop([523,1182],inplace=True)

y_train.drop([523,1182],inplace=True)

x_train.sort_values(by = 'GarageArea', ascending = False)[:4]

x_train.drop([581,1190,1061,825],inplace=True)

y_train.drop([581,1190,1061,825,1298],inplace=True)

print(x_train.shape,y_train.shape)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=5,min_samples_split=3,random_state=1)

model.fit(x_train,y_train)
x_test=test[features]

x_test = x_test.fillna(x_train.mean())

predictions = model.predict(x_test)

output=pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})

output.to_csv('submission.csv',index=False)

output