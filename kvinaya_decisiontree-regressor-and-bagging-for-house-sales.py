# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from datetime import datetime,date,time

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import BaggingRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/kc_house_data.csv")

data.head()

data.shape# find number of rows and columns 
data.isnull().sum() 
data.info()
data["date"]=pd.to_datetime(data["date"])

data["month"]=data["date"].dt.month

data["year"]=data['date'].dt.year

current_year=datetime.now().year

data["house_age"]=current_year-data["yr_built"] #create new colums for house_age
data=data.drop(["id",'date'],axis=1)

data.head()
sns.distplot(data.price)
a=data.ix[data["year"]==2015]['month'].value_counts()

a
b=data.ix[data["year"]==2014]['month'].value_counts()

b
a=data.groupby(["year",'month'])["month"].count().unstack("year")

ax = a.plot(kind='bar', stacked=True, alpha=0.7)

ax.set_xlabel('month', fontsize=14)

ax.set_ylabel('count', fontsize=14)

plt.xticks(rotation=0)

plt.show()
price_month=data['price'].groupby(data['month']).mean()

price_month.plot(kind='line')

plt.show()
#price difference between february & April

price_difference=price_month.max()-price_month.min()

price_difference

corr=data.corr()

corr.nlargest(24,'price')['price']
#df correlation matrix

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corr, annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()

labels=[u'bedrooms', u'bathrooms', u'sqft_living',

       u'sqft_lot', u'floors', u'waterfront', u'view', u'condition', u'grade',

       u'sqft_above', u'sqft_basement', u'yr_built', u'yr_renovated',u'lat', u'long', u'sqft_living15', u'sqft_lot15']

for i in range(len(labels)):

    plt.figure()

    sns.regplot(x=data[labels[i]],y="price",data=data);

    plt.xlabel(labels[i])

    plt.ylabel('price')

    plt.show()
X=data[[u'sqft_living',u'sqft_above',u'house_age',

        u'lat', u'long', u'sqft_living15',u'zipcode',u'sqft_lot15',u'waterfront', u'condition', u'grade']]

Y=data[['price']]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)

Y_train.shape
model = LinearRegression()

model.fit(X_train, Y_train)

train_score=model.score(X_train,Y_train)

train_score

test_score=model.score(X_test,Y_test)

test_score
#Decisiontree Regressor,BaggingRegressor

model = BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=10), n_estimators=5,bootstrap=True, bootstrap_features=False, oob_score=True, random_state=2, verbose=1).fit(X_train, Y_train)

test_score=model.score(X_test,Y_test)

train_score=model.score(X_train,Y_train)

train_score
test_score
from sklearn.model_selection import cross_val_score

score=cross_val_score(model,X,Y,cv=2)

score
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor(max_depth=6,random_state=5)

model.fit(X_train,Y_train)

predict=model.predict(X_test)

predict
score=model.score(X_train,Y_train)

score

score=model.score(X_test,Y_test)

score