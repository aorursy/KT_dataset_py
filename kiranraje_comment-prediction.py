# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/prediction-facebook-comment/Dataset.csv')

df.head()
#show data last 5 rows

df.tail()
#Describe function includes analysis of all our numerical data. For this, count, mean, std, min,% 25,% 50,% 75%, max values are given.

df.describe()
df.iloc[:,1:5].describe()
#Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.

df.isnull().sum()
#As you can see, most of the shares,mon_pub,thu_pub,mon_base value is empty. That's why I want this value deleted.

df=df.drop(['shares'],axis='columns')

df=df.drop(['mon_pub'],axis='columns')

df=df.drop(['thu_pub'],axis='columns')

df=df.drop(['mon_base'],axis='columns')
#filling the remaining null values

df['Returns']=df['Returns'].fillna(df['Returns'].mode()[0])

df['Category']=df['Category'].fillna(df['Category'].mode()[0])

df['commBase']=df['commBase'].fillna(df['commBase'].mode()[0])

df['comm48']=df['comm48'].fillna(df['comm48'].mode()[0])

#for checking the null values in given Dataset

df.isnull().any().any()
#scatter plot

import matplotlib.pyplot as plt

plt.scatter(df.sat_base,df.output,marker='+',color='red')

plt.xlabel('sat_base')

plt.ylabel('output')
plt.scatter(df.comm24,df.output,marker='+',color='red')

plt.xlabel('comm24')

plt.ylabel('output')
plt.scatter(df.baseTime,df.output,marker='+',color='red')

plt.xlabel('baseTime')

plt.ylabel('output')
import seaborn as sns

def correlation_heatmap(df):

    _,ax=plt.subplots(figsize=(20,20))

    colormap=sns.diverging_palette(220,10,as_cmap=True)

    sns.heatmap(df.corr(),annot=True,cmap=colormap)

    

    

correlation_heatmap(df)
#plot histogram of each parameter

import matplotlib.pyplot as plt

df.hist(figsize=(20,20))

plt.show()
print(df.columns)
#get all the columns from the dataframe

columns=df.columns.tolist()



#filter the columns to remove data we do not want

columns=[c for c in columns if c not in ['output']]



#store the value we will predicting on

target='output'



x=df[['likes', 'Checkins', 'Returns', 'Category', 'commBase', 'comm24',

       'comm48', 'comm24_1', 'diff2448', 'baseTime', 'length', 'hrs',

       'sun_pub', 'tue_pub', 'wed_pub', 'fri_pub', 'sat_pub', 'sun_base',

       'tue_base', 'wed_base', 'thu_base','fri_base']]

y=df[target]



#print the shape of x and y

print(x.shape)

print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
len(x_train)
len(x_test)
from sklearn import tree
model=tree.DecisionTreeRegressor()

model.fit(x_train,y_train)
model.predict(x_test)
model.score(x_test,y_test)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=20)

model.fit(x_train, y_train)
model.predict(x_test)
model.score(x_test,y_test)