import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
%matplotlib inline
matplotlib.rcParams["figure.figsize"] =(20,10)
!ls
df = pd.read_csv('../input/housing-in-london/housing_in_london_monthly_variables.csv')
df.head()
df.describe()
df.shape
df.isna().sum()
#droping the no_of_crimes column as it has no meaningful data
df1 = df.drop(['no_of_crimes'],axis='columns')
df1.head()
#checking if we have NA values
df1.isna().sum()
df1.shape
#droping the 94 columns of houses sold na values
df2 = df1.dropna()
df2.isna().sum()
df2.shape
df2.head()
import datetime
#checking the type of date, it is an object i.e string which needs to be converted to datetime
df2['date'].dtype
#checking the type of all the columns
df2.dtypes
#Copying the dataframe into a new df
df3 = df2.copy()
df3.shape
df2.shape
#Converting the string ino the datetime
df3['datetime'] = pd.to_datetime(df3['date']) 
df3.head()
df3.dtypes
#dropping the date with string values.
df3.drop(['date'],axis='columns')
#let us check what are the unique values for area
df3['area'].unique()
#let us check what are the unique values for area
df3['houses_sold'].unique()
matplotlib.rcParams["figure.figsize"] =(30,20)
matplotlib.pyplot.bar(df3['area'],df3['average_price'])
def which_city(str):
    if 'london' in str:
        return 'London'
    else:
        return 'Rest of England'

which_city('east london')
#feature engineering for a new feature called city
df3['city'] =  df3['area'].apply(lambda x: which_city(x))
df3.head()
matplotlib.pyplot.bar(df3['city'],df3['average_price'])
#new feature called year

df3['year'] =  df3['datetime'].apply(lambda x: x.year)
df3.head()
df3.drop(['date'],axis='columns')
df4 = df3.copy()
df5 = df4.groupby('year').sum()
df5.head()

df5['houses_sold'].plot(kind='bar')
df5['average_price'].plot(secondary_y=True)
df4.head()
df4.shape
df6 = df4.copy()
df6['revenue'] = df6['average_price']*df6['houses_sold']
df6.head()
rev = df6.groupby('year').sum()
rev.head()
rev['revenue'].plot(kind="bar")
import sklearn
df4.head()
df4.info()
#to run a ML we need to conver objects into non-object notations
df4['code'].unique()
df4['code'] = df.code.str.replace('E','').astype(float)
df4['code'].unique()
df5 = df4.drop(['date','borough_flag'],axis='columns')
df5['area'].unique()
area_dummies = pd.get_dummies(df5['area'])
df7 = pd.concat([df5,area_dummies],axis='columns')
df7.describe()
df8 = df7.drop(['area','city','year','datetime'],axis='columns')
#a town or district which is an administrative unit.
X=df8.drop(['average_price'],axis='columns')
y=df8[['average_price']]
X.dtypes
y.dtypes
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=1,)
from sklearn.linear_model  import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
#Accuracy is very low
