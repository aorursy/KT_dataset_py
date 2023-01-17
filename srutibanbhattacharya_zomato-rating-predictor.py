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
df=pd.read_csv('/kaggle/input/zomato-restaurants-data/zomato.csv',encoding='latin-1')
df.head(2)
df.shape
#predicting average cost for any given resturaunt
#we do not need resto id
#also drop resto na,e for now because it is difficutl to deal with....come back to it later
df['Restaurant Name'].value_counts()
df['Is delivering now'].value_counts()
#highly biased
#get rid of price range because its basically another ouput
#get rid of 'is delivering now' because lots of bias

df.drop(columns=['Restaurant ID','Restaurant Name','Is delivering now','Switch to order menu','Price range','Rating color'],axis=1,inplace=True)

df.drop(columns=['Address','Locality','Locality Verbose'],axis=1,inplace=True)
df.head(2)
df['Country Code'].value_counts()
df['Country Code'].value_counts().shape
# 90% resto from india ...therefore to feed algo only simplified data lets just consider only those where country code=1 ie. India.
# since only India therefore also removing currency
df[df['Country Code']==1]
df=df[df['Country Code']==1]
df.drop(columns=['Country Code','Currency'],axis=1,inplace=True)
df.head(2)
df['City'].value_counts()
#4 citys have 8k restos while rest have only a few... therefore removing them
df=df[df['City'].isin(['New Delhi','Gurgaon','Noida','Faridabad'])]
df.sample(5)
df['City'].value_counts()
#for now keep aside cuisines
#using label encoding for strings or assigning a number to categorical data

from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
#making object from class
df['City']=encoder.fit_transform(df['City'])
df['City'].value_counts()
df['Has Table booking']=encoder.fit_transform(df['Has Table booking'])
df['Has Online delivery']=encoder.fit_transform(df['Has Online delivery'])
df['Rating text']=encoder.fit_transform(df['Rating text'])
#to solve multi encoding problem so 3 is not given more weightage than..say 1 or delhi
#so just change them to columns from rows
# one hot encoding is what this is called
# then reomve multi-collinearity using get dummies

df=pd.get_dummies(df, columns=['City','Rating text'],drop_first=True)

df.head()
df.groupby('Cuisines').mean()
cuisine=df.groupby('Cuisines').mean()['Average Cost for two'].reset_index()
cuisine
#merging cuisine with earlier dataset df

df=df.merge(cuisine,on='Cuisines')
df

#mean value of cuisines we got so dropping original cuisines
df.drop(columns=['Cuisines'],axis=1,inplace=True)
df.rename(columns={'Average Cost for two_y':'Cuisines'},inplace=True)
df.head()
#now using correlation matrix
df.corr()
#corr function showing correlation b/w 'av price for two' and other columns#showing correlation b/w 'av price for two' and other columns
df.corr()['Average Cost for two_x']

#1:Extract X and Y

X=df.drop(columns=['Average Cost for two_x']).values
Y=df['Average Cost for two_x'].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
reg.predict(X_test)

#algo giving predicted prices for all the resturaunts 
#now storing the pred values in variable  Y_pred
Y_pred=reg.predict(X_test)
print (X_test.shape)
print(Y_pred.shape)
print (Y_test.shape)
#now we have to compare Y_test with Y_pred to determine efficacy of the model
Y_pred[2]
Y_test[2]
#Now we find r2 score to determine how well model is working
from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)
# it is found that the regression is around 73% accurate
