# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/CarPrice_Assignment.csv')
df.head()
## Seperate the compay name from the car name

df['company']=df.CarName.apply(lambda x:x.split()[0])
# I wrote function to make car name to just the name of the model

def WordReplace(x,y):

    return x.replace(y+' ','')
# use the WordReplace to convert carname to just the model name

df['CarName']=df.apply(lambda x:WordReplace(x.CarName,x.company),axis=1)
# set car_id as the index of the table, if you add df.head(), we can see the new 

df.set_index('car_ID',inplace=True);
# Just to check who is producing more models, looks like Toyota leads the board with 30 models

df['company'].value_counts().plot('bar',figsize=(10,12))
def NameToNumber(x):

    return {'four':4,'six':6,'five':5,'eight':8,'two':2,'twelve':12,'three':3}[x]
fig,ax=plt.subplots(2,1,figsize=(16,16))

s=df['company'].value_counts()

s[s>4].index

#print(s[s>4].index)

sns.boxplot(x='company',y='enginesize',data=df[df.company.isin(s[s>5].index)],ax=ax[0])

sns.boxplot(x='company',y='price',data=df[df.company.isin(s[s>5].index)],ax=ax[1])



## so the saab,toyota,and volvo use the same engine for the all the cars in the country

## audi,dodge,plymoth and VW also seems to use the same engine with small variation

## bmw,buck,nissan,toyota uses a very wide range of engines

# change the cylinder number from categorical to numerical



df['cylindernumber']=df.cylindernumber.apply(lambda x:NameToNumber(x))





# Delete the following columns as they won't have any categories



df.drop(columns=['symboling','CarName','aspiration','enginetype','fuelsystem'],inplace=True)

corr=df.corr(method='pearson')

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





# getting columns with correlation greater than 0.6

print(corr.loc['price',np.absolute(corr.loc['price',:])>0.6])



# columns with the numerical datas

p=corr.loc['price',np.absolute(corr.loc['price',:])>0.6].keys()

s=df.dtypes



# Getting columns whose datatype is categorical

print('columns with categorical data',s[s=='object'].keys())



# get dummy variables for the categories

df_categories=pd.get_dummies(df[s[s=='object'].keys()])
# Drop the categorical columns from teh df as the dummy variables are created

df=df.drop(columns=(s[s=='object'].keys()),axis=1)
# create the dataframe numerical columns



df_numbers=df[p]



# create the df2 with the set of numerical and category datasframes



df2=df_numbers.join(df_categories)
df2.head()
X=df2.drop(columns=['price'],axis=1).values

Y=df2.price.values
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.1, random_state=42,shuffle=True)
pipeline=Pipeline([('scaler',MinMaxScaler()),('Regressor',LinearRegression())])
pipeline.fit(X_train,y_train)

scores=cross_val_score(estimator=pipeline,X=X_train,y=y_train,cv=5,scoring='r2')
print(scores)

print('mean score:',scores.mean())
# Testing on data which the model has not seen before

r2_score(y_test,pipeline.predict(X_test))