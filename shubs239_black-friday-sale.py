# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/BlackFriday.csv')
df.head()
df.info()
df['Product_Category_2']
sns.barplot(x='Marital_Status',y='Purchase',data=df,hue='Gender')
df.describe()
sns.barplot(x='Age',y='Purchase',data=df,hue='Gender')
sns.lmplot(x='Product_Category_2',y='Purchase',data=df)
plt.figure(figsize=(12,6))
sns.barplot(x='Product_Category_2',y='Purchase',data=df,hue='Gender')
plt.figure(figsize=(12,6))
sns.barplot(x='Product_Category_2',y='Purchase',data=df,hue='Marital_Status')
plt.figure(figsize=(12,6))
sns.barplot(x='Product_Category_2',y='Purchase',data=df,hue='City_Category')
plt.figure(figsize=(12,6))
sns.barplot(x='Product_Category_3',y='Purchase',data=df,hue='City_Category')
df[df['City_Category']=='C']['Product_Category_3'].mean()
df['Product_Category_3'].hist(bins=30)
df['Product_Category_2'].value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x='Age',data=df)
df[df['Age']=='26-35']['Product_Category_3'].hist(bins=20)
df[df['Age']=='26-35']['Product_Category_2'].mean()
p2a=df[df['City_Category']=='A']['Product_Category_2'].mean()
int(p2a)
p2b=df[df['City_Category']=='B']['Product_Category_2'].mean()
int(p2b)

##Since most of the City C users are using Product 2 and Product 3, I am replacing NaN values by the mean of those values
p2c=df[df['City_Category']=='C']['Product_Category_2'].mean()
int(p2c)
p3a=df[df['City_Category']=='C']['Product_Category_3'].mean()
int(p3a)
df['Product_Category_3'].fillna(12,inplace=True)
df.head()
data=df.drop(['User_ID','Product_ID'],axis=1)
sex=pd.get_dummies(data['Gender'],drop_first=True)
age=pd.get_dummies(data['Age'])
city=pd.get_dummies(data['City_Category'],drop_first=True)
real_data=data.join([sex,age,city])
real_data.head()
real_data.replace(['0','1','2','3','4+'],[0,1,2,3,5],inplace=True)
real_data.head()
real_data['Stay_In_Current_City_Years'].unique()
real_data.head()

real_data.drop('City_Category',axis=1,inplace=True)

real_data.head()
X=real_data.drop('Purchase',axis=1)
y=real_data['Purchase']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from xgboost import XGBRegressor
xgb=XGBRegressor(n_estimators=1000,learning_rate=0.05)
xgb.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[(X_test,y_test)],verbose=False)
predictions=xgb.predict(X_test)
from sklearn.metrics import mean_absolute_error
print(np.sqrt(mean_absolute_error(y_test,predictions)))

















