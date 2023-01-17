import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/train.csv')
df.head()
df.describe()
1
df.fillna(value=0,inplace=True)
df.isnull().sum()
df.dtypes
df['Gender'].nunique()
Gender=pd.get_dummies(df['Gender'],drop_first=True)

df=pd.concat([df,Gender],axis=1)

df.drop('Gender',axis=1,inplace=True)
df.head()
df['Stay_In_Current_City_Years'].unique()
def stay(x):

    if x=='4+':

        return 4

    else:

        return int(x)

df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].apply(stay)
df['Age'].unique()
def age(x):

    if x=='0-17':

        return 0

    elif x=='18-25':

        return 1

    elif x=='26-35':

        return 2

    elif x=='36-45':

        return 3

    elif x=='46-50':

        return 4

    elif x=='51-55':

        return 5

    else:

        return 6

df['Age']=df['Age'].apply(age)
df.head()
df.drop(['User_ID','Product_ID'],axis=1,inplace=True)
df.head()
def city(x):

    if x=='A':

        return 0

    elif x=='B':

        return 1

    else :

        return 2

df['City_Category']=df['City_Category'].apply(city)
df.head()
df.dtypes
def con(x):

    return int(x)

df['Product_Category_2']=df['Product_Category_2'].apply(con)

df['Product_Category_3']=df['Product_Category_3'].apply(con)

df.head()
from sklearn.model_selection import train_test_split
x=df[['Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','M']]
y=df['Purchase']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(x_train,y_train)
print(lm.intercept_)
lm.coef_
predictions=lm.predict(x_test)
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
y_test
predictions