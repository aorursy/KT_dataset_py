import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df=pd.read_csv('../input/kc_house_data.csv')

print([df.columns])

df.info()
def miss(x):

    return(sum(x.isnull()))

df.apply(miss)
fig, ax = plt.subplots(figsize=(12,12)) 

sns.heatmap(df.corr(), annot=True)
df
df['year_sell'] = [int(i[:4]) for i in df.date]



df.year_sell.value_counts().sort_index()
df['price_per_sqft'] = df['price']/df['sqft_living']# square feet_living



mean =df['price_per_sqft'].mean()
B =plt.boxplot('price_per_sqft',data=df)

l =[item.get_ydata()[0] for item in B['whiskers']]
l[1]
fig, ax = plt.subplots(figsize=(10,10)) 

plt.scatter(df['lat'],df['long'],alpha=0.33)

plt.scatter(df[(df.price_per_sqft> mean)&(df.price_per_sqft< l[1])].lat,df[(df.price_per_sqft> mean)&(df.price_per_sqft< l[1])].long)

plt.scatter(df[df.price_per_sqft> l[1]].lat,df[df.price_per_sqft> l[1]].long)

df['age_of_renov'] = 100

df.loc[df['yr_renovated'] != 0,'age_of_renov'] = 2015-df.loc[df['yr_renovated'] != 0,'yr_renovated']
sns.regplot(x='age_of_renov',y='price',data=df)
sns.regplot('sqft_living','price',data=df)
sns.regplot('sqft_lot','price',data=df)
sns.regplot('sqft_above','price',data=df)
df['area_floor'] =df['sqft_above']+df['sqft_living']
sns.regplot('area_floor','price',data=df)
df.columns
from sklearn.linear_model import LinearRegression,Ridge, Lasso
df['basement_present'] = df['sqft_basement'].apply(lambda x: 1 if x > 0 else 0) # Indicate whether there is a basement or not

df['renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0) # 1 if the house has been renovated
categorial_cols = ['floors', 'view', 'condition', 'grade']



for cc in categorial_cols:

    dummies = pd.get_dummies(df[cc], drop_first=False)

    dummies = dummies.add_prefix("{}_".format(cc))

    df.drop(cc, axis=1, inplace=True)

    df = df.join(dummies)
df.columns
col = ['bedrooms', 'bathrooms', 'sqft_living','basement_present',

       'sqft_lot', 'waterfront',

       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',

        'sqft_living15', 'sqft_lot15','price_per_sqft', 'age_of_renov', 'area_floor','floors#1.0', 'floors#1.5', 'floors#2.0', 'floors#2.5', 'floors#3.0',

       'floors#3.5', 'view#0', 'view#1', 'view#2', 'view#3', 'view#4',

       'condition#1', 'condition#2', 'condition#3', 'condition#4',

       'condition#5', 'grade#1', 'grade#3', 'grade#4', 'grade#5', 'grade#6',

       'grade#7', 'grade#8', 'grade#9', 'grade#10', 'grade#11', 'grade#12',

       'grade#13']
test=df[df.year_sell==2015].reset_index(drop=True )

train=df[df.year_sell==2014].reset_index(drop=True )

x=train[df.columns.drop(['price','id','date'])]

y=train['price']
col=df.columns.drop(['price','id','date'])

clf = LinearRegression()

clf.fit(x,y)
clf.score(x,y)
a=[0.0001,0.0003,0.001,0.005,0.1,0.5,1]

for i in a:   

    print(i)

    clf3= Lasso(alpha=i)

    clf3.fit(x,y)

    print(clf3.score(x,y))
pred3 = clf3.predict(test[col])
pred = clf.predict(test[col])
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

x=train[col]

y=train['price']

clf=RandomForestRegressor()

clf2=XGBRegressor()

clf2.fit(x,y)
clf2.score(x,y)
pred2 = clf2.predict(test[col])
sum((test['price']-pred2)*(test['price']-pred2))/len(pred2)
sum((test['price']-pred)*(test['price']-pred))/len(pred)
sum((test['price']-pred3)*(test['price']-pred3))/len(pred3)