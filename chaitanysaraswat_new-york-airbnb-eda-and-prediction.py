

import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

import geopandas

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn import metrics

sns.set_style('darkgrid')
df=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.info()
df.dtypes
df.describe()
df.shape
for col in df[['neighbourhood_group','neighbourhood','last_review', 'host_name','room_type']]:

        print('unique values %s=' %col)

        print(df[col].unique()), '\n'
print('null values in airbnb dataset')

print(df.isnull().sum())

hostid=(pd.DataFrame((np.where(df['name'].isnull())))).transpose()

hostid
hostname=pd.DataFrame(np.where(df['host_name'].isnull())).transpose()

hostname

null=pd.concat([hostid,hostname],axis=1,ignore_index=True)

null
df.drop('last_review',axis=1,inplace=True)

df
df['name'].fillna(value=0,inplace=True)

df.isnull().sum()
df['host_name'].fillna(value=0,inplace=True)

df.isnull().sum()
df['reviews_per_month'].fillna(value=0,inplace=True)

df.isnull().sum()
def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b=pandas_obj.memory_usage(deep=True).sum()

    else:

        usage_b=pandas_obj.memory_usage(deep=True)

    usage_mb=usage_b/1024**2

    return "{:03.2f} MB".format(usage_mb)

        

mem_usage(df)
plt.figure(figsize=(15,6))

sns.countplot(data=df,x='neighbourhood_group',hue='room_type',palette='Blues')

plt.title('neighbourhood group with room types they alot')

plt.show()

plt.figure(figsize=(17,5))

sns.countplot(x='room_type',data=df,palette='Blues')

plt.title('rooms that are most favourable for customer')

plt.show()
plt.figure(figsize=(15,6))

sns.violinplot(data=df[df.price <500], x='neighbourhood_group', y='price', palette='GnBu_d')

plt.show()
plt.figure(figsize=(15,8))

sns.heatmap(df.corr(),annot=True,linewidth=0.1,cmap='Blues')
plt.figure(figsize=(10,10))

sns.scatterplot(data=df,x='longitude',y='latitude',hue='neighbourhood_group',palette='Blues')

plt.show()
df.head()
df.drop(['id','name','host_name'],inplace=True,axis=1)

df
df.isnull().sum()
df.head(10)
le=preprocessing.LabelEncoder()



le.fit(df['neighbourhood_group'])

df['neighbourhood_group']=le.transform(df['neighbourhood_group'])



le.fit(df['neighbourhood'])

df['neighbourhood']=le.transform(df['neighbourhood'])



le.fit(df['room_type'])

df['room_type']=le.transform(df['room_type'])











df.sort_values('price',ascending=True,inplace=True)

df=df[11:-6]



lm=LinearRegression()
X=df.drop(['price','longitude'] ,inplace=False,axis=1)

y=df['price']

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=101)

lm.fit(X_train,y_train)
prediction=lm.predict(X_test)

prediction
mae=metrics.mean_absolute_error(y_test,prediction)

mse=metrics.mean_squared_error(y_test,prediction)

rmse=np.sqrt(metrics.mean_squared_error(y_test,prediction))

r2=metrics.r2_score(y_test,prediction)



print( mae,mse,rmse,r2)
error=pd.DataFrame({'Actual Values': np.array(y_test).flatten(),'predicted values':prediction.flatten()})

error.head(10)
plt.figure(figsize=(20,6))

plt.xlim(-10,350)

sns.regplot(y=y_test,x=prediction,color='blue')
