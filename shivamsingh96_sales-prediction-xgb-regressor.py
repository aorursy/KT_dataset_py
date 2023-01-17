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
train = pd.read_csv("../input/black-friday/train.csv")
test = pd.read_csv("../input/black-friday/test.csv")
train.head()
train.shape
train.info()
train.isnull().sum()
train.describe()
missing_values = train.isnull().sum().sort_values(ascending = False)
missing_values = missing_values[missing_values > 0]/train.shape[0]
print(f'{missing_values *100} %')
train['Product_Category_3'].unique()
train['Product_Category_2'].unique()
test.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
train['Age'].value_counts()
plt.figure(figsize=(10,5))
sns.countplot(train['Age'])
plt.figure(figsize=(10,5))
sns.countplot(train['City_Category'])
train['Occupation'].unique()
train['City_Category'].unique()
train['Stay_In_Current_City_Years'].unique()
plt.figure(figsize=(15,8))
sns.boxplot(train['Product_Category_1'],train['Purchase'])
plt.figure(figsize=(15,8))
sns.boxplot(train['Age'],train['Purchase'])
plt.figure(figsize=(10,5))
sns.countplot(train['City_Category'])
plt.figure(figsize=(15,8))
sns.boxplot(train['City_Category'],train['Purchase'])
plt.figure(figsize=(15,8))
sns.boxplot(train['Stay_In_Current_City_Years'],train['Purchase'])
plt.figure(figsize=(10,5))
sns.distplot(train['Purchase'],color='green')
def label_encoding(df):
    df['Age']=df['Age'].replace('0-17',17)
    df['Age']=df['Age'].replace('18-25',25)
    df['Age']=df['Age'].replace('26-35',35)
    df['Age']=df['Age'].replace('36-45',45)
    df['Age']=df['Age'].replace('46-50',50)
    df['Age']=df['Age'].replace('51-55',55)
    df['Age']=df['Age'].replace('55+',60)
    df['Gender']=df['Gender'].replace('F',0)
    df['Gender']=df['Gender'].replace('M',1)
    df['City_Category']=df['City_Category'].replace('A',0)
    df['City_Category']=df['City_Category'].replace('B',1)
    df['City_Category']=df['City_Category'].replace('C',2)
    df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].replace('4+',4)
    return df
train=label_encoding(train)
test=label_encoding(test)
train['Stay_In_Current_City_Years']=train['Stay_In_Current_City_Years'].astype(int)
test['Stay_In_Current_City_Years']=test['Stay_In_Current_City_Years'].astype(int)
train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
test.isnull().sum()
train.isnull().sum()
dfd=train.copy()
dft=test.copy()
avg_purchase_per_product=pd.DataFrame(train.groupby(['Product_ID'])['Purchase'].mean())
avg_purchase_per_product.reset_index(inplace=True)
avg_purchase_per_user=pd.DataFrame(train.groupby(['User_ID'])['Purchase'].mean())
avg_purchase_per_user.reset_index(inplace=True)
def create_var(data):
    product_count=pd.DataFrame(data['Product_ID'].value_counts())

    product_count.reset_index(inplace=True)
    product_count=product_count.rename(columns={'index':'Product_ID','Product_ID':'Product_count'})

    data['avg_purchase_per_product']=data['Product_ID'].map(avg_purchase_per_product.set_index('Product_ID')['Purchase'])
    data['product_count']=data['Product_ID'].map(product_count.set_index('Product_ID')['Product_count'])
    data['avg_purchase_per_user']=data['User_ID'].map(avg_purchase_per_user.set_index('User_ID')['Purchase'])

    conditions = [
    (data['Product_Category_1'] != 0) & (data['Product_Category_2'] == 0) & (data['Product_Category_3'] == 0),
    (data['Product_Category_1'] != 0) & (data['Product_Category_2'] != 0) & (data['Product_Category_3'] == 0),
    (data['Product_Category_1'] != 0) & (data['Product_Category_2'] != 0) & (data['Product_Category_3'] != 0)]
    choices = [1, 2, 3]
    data['Category_Count'] = np.select(conditions, choices, default=0)
    
    return data
train=create_var(train)
test=create_var(test)
train['Product_ID']=train['Product_ID'].str.slice(2).astype(int)
test['Product_ID']=test['Product_ID'].str.slice(2).astype(int)
train.head()
test.head()
test.fillna(0,inplace=True)
corr=train.corr()
plt.figure(figsize=(20,12))
sns.heatmap(corr,annot=True)
gender_p=train.groupby(['Gender'])['Purchase'].mean()
age_p=train.groupby(['Age'])['Purchase'].mean()
occupation_p=train.groupby(['Occupation'])['Purchase'].mean()
gender_p
age_p
occupation_p
print(occupation_p.mean())
print(occupation_p.std())
city_cat_p=train.groupby(['City_Category'])['Purchase'].mean()
city_cat_p
marital_p=train.groupby(['Marital_Status'])['Purchase'].mean()
marital_p
years_p=train.groupby(['Stay_In_Current_City_Years'])['Purchase'].mean()
years_p
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost.sklearn import XGBRegressor
X=train.drop('Purchase',axis=1)
y=train['Purchase']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)
reg=linear_model.LinearRegression()
lm_model=reg.fit(X_train,y_train)
pred=lm_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,pred))
RF_reg=RandomForestRegressor()
RF_model=RF_reg.fit(X_train,y_train)
np.sqrt(mean_squared_error(y_test,pred))
xgb=XGBRegressor()
XGB_model=xgb.fit(X_train,y_train)
xg_pred=XGB_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,xg_pred))
