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

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.pandas.set_option('display.max_rows',None)
pd.pandas.set_option('display.max_columns',None)
train=pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
train.head()
train.shape
train.head()
target=train['Category'].unique()
print(target)
data_dict = {}
count = 1
for data in target:
    data_dict[data] = count
    count+=1
    train["Category"] = train["Category"].replace(data_dict)
train.head(50)
x=train['DayOfWeek'].unique()
print(x)
data_week_dict={
    'Monday':1,
    'Tuesday':2,
    'Wednesday':3,
    'Thursday':4,
    'Friday':5,
    'Saturday':6,
    'Sunday':7
}
train['DayOfWeek']=train['DayOfWeek'].replace(data_week_dict)
test=pd.read_csv('/kaggle/input/sf-crime/test.csv.zip')
test.head()
z=test['Id']
test.shape
test['DayOfWeek']=test['DayOfWeek'].replace(data_week_dict)
train.drop(['Resolution','Descript'],axis=1,inplace=True)
test.drop(['Id'],axis=1,inplace=True)

train.head()
test.head()
test_x=test.copy()
train_x=train.copy()
dataset=pd.concat((train,test),axis=0)
dataset.columns
dataset['Dates'].head()
dataset['Dates']=pd.to_datetime(dataset['Dates'],format='%Y-%m-%d %H:%M:%S',errors='coerce')
dataset['Year']=dataset['Dates'].dt.year
dataset['Month']=dataset['Dates'].dt.month
dataset['Date']=dataset['Dates'].dt.day
dataset['Hour']=dataset['Dates'].dt.hour
dataset['Minutes']=dataset['Dates'].dt.minute
dataset.head()
dataset.drop(['Dates'],axis=1,inplace=True)
dataset.head()
dataset.isnull().sum()
numerical_features=[feature for feature in dataset.columns if dataset[feature].dtypes!='O']
numerical_features
dataset[numerical_features].head()
categorical_features=[feature for feature in dataset.columns if dataset[feature].dtypes=='O']
categorical_features
dataset[categorical_features].head()
dataset[categorical_features].columns
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for feature in ['PdDistrict', 'Address']:
    dataset[feature]=le.fit_transform(dataset[feature])
dataset[categorical_features].head()
dataset.head()
core=dataset.corr()
core
plt.figure(figsize=(20,20))
sns.heatmap(core,annot=True)
corr=dataset.corr()
print(corr['Category'])
dataset.head()
train=dataset.iloc[:878049,:]
train.head()
train.shape
test=dataset.iloc[878049:]
test.head()
test.drop(['Category'],axis=1,inplace=True)
test.head()
test=pd.concat((z,test),axis=1)
test.head()
test.shape
X_train=train.drop(['Category'],axis=1)
y_train=train['Category']
X_test=test
X_train['DayOfWeek'].unique()
X_test.head(100)
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
X_test.head()
X_test.drop(['Id'],axis=1,inplace=True)
predictions=knn.predict(X_test)
from collections import OrderedDict
data_dict_new = OrderedDict(sorted(data_dict.items()))
print(data_dict_new)
predictions
test=pd.concat((z,test),axis=1)
result_dataframe = pd.DataFrame({
    "Id": test["Id"]
})
for key,value in data_dict_new.items():
    result_dataframe[key] = 0
count = 0
for item in predictions:
    for key,value in data_dict.items():
        if(value == item):
            result_dataframe[key][count] = 1
    count+=1
result_dataframe.to_csv("submission_knn.csv", index=False)

