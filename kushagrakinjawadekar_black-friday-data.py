# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/black-friday'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
train = pd.read_csv('/kaggle/input/black-friday/train.csv')

test = pd.read_csv('/kaggle/input/black-friday/test.csv')
train.head()
train.info()
train.describe()
train.isnull().sum()
import seaborn as sns

sns.heatmap(train.isnull(), cbar=False)
sns.set(rc = {'figure.figsize':(16,10)})

sns.heatmap(train.corr(),annot=True,cbar=False)
plt.figure(figsize=(10,5))

sns.countplot(train['Age'])

plt.figure(figsize=(10,5))

sns.countplot(train['Stay_In_Current_City_Years'])
train.head()
plt.figure(figsize=(10,5))

sns.countplot(train['City_Category'])
train[['Product_ID','Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status']].nunique()
sns.distplot(train['Product_Category_3'])
sns.distplot(train['Product_Category_2'])
train['Product_Category_2'].fillna(train['Product_Category_2'].mean(),inplace=True)

train['Product_Category_3'].fillna(train['Product_Category_3'].mean(),inplace=True)

test['Product_Category_2'].fillna(train['Product_Category_2'].mean(),inplace=True)

test['Product_Category_3'].fillna(train['Product_Category_3'].mean(),inplace=True)

train.isnull().sum()
train['Stay_In_Current_City_Years'].replace({'4+':4},inplace=True)

test['Stay_In_Current_City_Years'].replace({'4+':4},inplace=True)
# Gender

train['Gender'].replace({"M":1,"F":0},inplace=True)

test['Gender'].replace({"M":1,"F":0},inplace=True)
# Age

def map_age(age):

    if age == '0-17':

        return 0

    elif age == '18-25':

        return 1

    elif age == '26-35':

        return 2

    elif age == '36-45':

        return 3

    elif age == '46-50':

        return 4

    elif age == '51-55':

        return 5

    else:

        return 6
train['Age'] = train['Age'].apply(map_age)

test['Age'] = test['Age'].apply(map_age)
# Mapping the City_Category 



train['City_Category']=train['City_Category'].map({"B":1,"A":2,"C":3})

test['City_Category']=test['City_Category'].map({"B":1,"A":2,"C":3})
test['City_Category']= test['City_Category'].astype(int)

train['City_Category']= train['City_Category'].astype(int)
train['Stay_In_Current_City_Years']= train['Stay_In_Current_City_Years'].astype(int)

test['Stay_In_Current_City_Years']= test['Stay_In_Current_City_Years'].astype(int)
train = train.drop(["User_ID","Product_ID"],axis=1)
from sklearn.model_selection import train_test_split

X = train.drop("Purchase",axis=1)

y = train['Purchase']





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)



import xgboost as xgb

from sklearn.metrics import mean_squared_error



xg=xgb.XGBRegressor()

xg.fit(X_train, y_train)

xg_predict= xg.predict(X_test)
print("RMSE score for XGB Regressor : ", np.sqrt(mean_squared_error(y_test,xg_predict)))
test.head()
X_testing = test[['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1'

                 ,'Product_Category_2','Product_Category_3']]
predict = xg.predict(X_testing)
Submit = pd.DataFrame({'Purchase':predict,'User_ID':test['User_ID'],'Product_ID':test['Product_ID']})
Submit.to_csv('submission.csv',index=False)