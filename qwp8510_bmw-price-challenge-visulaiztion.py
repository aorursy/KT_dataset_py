%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df = pd.read_csv('../input/bmw_pricing_challenge.csv')

df.head()
df.info()
df.describe()
sns.set(style='darkgrid')

sns.countplot(x='fuel',data=df,palette='Set2')
df['fuel'].value_counts()
sns.set(style='darkgrid')

sns.countplot(x='car_type',data=df,palette='Set2')
plt.figure(figsize=(20,10))

sns.boxplot('car_type','engine_power',data=df)
plt.figure(figsize=(20,10))

sns.boxplot('car_type','price',data=df)
sns.catplot(x='car_type',y='mileage',hue='fuel',data=df,kind='swarm',split=True)
df_sold_at = df.groupby(['sold_at']).first()

df_sold_at
df_sold_at.count()
df_estate = df[df.car_type=='estate']

df_sedan = df[df.car_type=='sedan']

df_suv = df[df.car_type=='suv']
df_estate.head()
plt.figure(figsize=(20,10))

sns.countplot('model_key',data=df_estate,palette='bright')
plt.figure(figsize=(20,10))

sns.countplot('engine_power',data=df_estate,palette='bright')
sns.countplot('paint_color',data=df_estate,palette='bright')
plt.figure(figsize=(20,10))

sns.barplot('model_key','price',data=df_estate)
plt.figure(figsize=(20,10))

sns.boxplot('model_key','mileage',data=df_estate)
plt.scatter('mileage','price',data=df_estate)

plt.xlabel('mileage of estate')

plt.ylabel('price of estate')
plt.figure(figsize=(20,10))

d = sns.countplot('model_key',data=df_sedan,palette='bright')

_ = plt.setp(d.get_xticklabels(),rotation=90)
plt.figure(figsize=(20,10))

sns.countplot('engine_power',data=df_sedan,palette='bright')
plt.figure(figsize=(20,10))

d = sns.barplot('model_key','price',data=df_sedan)

_ = plt.setp(d.get_xticklabels(),rotation=90)
plt.figure(figsize=(20,10))

d = sns.boxplot('model_key','mileage',data=df_sedan)

_ = plt.setp(d.get_xticklabels(),rotation=90)
plt.scatter('mileage','price',data=df_sedan)

plt.xlabel('mileage of sedan')

plt.ylabel('price of sedan')
d = sns.countplot('model_key',data=df_suv,palette='bright')
plt.figure(figsize=(20,10))

sns.countplot('engine_power',data=df_suv,palette='bright')
d = sns.barplot('model_key','price',data=df_suv)
d = sns.boxplot('model_key','mileage',data=df_suv)
plt.scatter('mileage','price',data=df_suv)

plt.xlabel('mileage of suv')

plt.ylabel('price of suv')
df['car_type'] =df.car_type.map(lambda x:1.0 if x =='convertible' else x)

df['car_type'] =df.car_type.map(lambda x:2.0 if x =='coupe' else x)

df['car_type'] =df.car_type.map(lambda x:3.0 if x =='estate' else x)

df['car_type'] =df.car_type.map(lambda x:4.0 if x =='hatchback' else x)

df['car_type'] =df.car_type.map(lambda x:5.0 if x =='sedan' else x)

df['car_type'] =df.car_type.map(lambda x:6.0 if x =='suv' else x)

df['car_type'] =df.car_type.map(lambda x:7.0 if x =='van' else x)

df['car_type'] =df.car_type.map(lambda x:8.0 if x =='subcompact' else x)

df['car_type'] = df['car_type'].astype(int)
df['feature_1'] =df.feature_1.map(lambda x:1 if x ==True else 0)

df['feature_2'] =df.feature_2.map(lambda x:1 if x ==True else 0)

df['feature_3'] =df.feature_3.map(lambda x:1 if x ==True else 0)

df['feature_4'] =df.feature_4.map(lambda x:1 if x ==True else 0)

df['feature_5'] =df.feature_5.map(lambda x:1 if x ==True else 0)

df['feature_6'] =df.feature_6.map(lambda x:1 if x ==True else 0)

df['feature_7'] =df.feature_7.map(lambda x:1 if x ==True else 0)

df['feature_8'] =df.feature_8.map(lambda x:1 if x ==True else 0)

df.head()
df.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor 

from sklearn.metrics import r2_score

from sklearn import metrics
x = df[['mileage','engine_power','feature_1','feature_2','feature_3','feature_4','feature_5','feature_6','feature_7','feature_8','car_type']]

y = df['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=36)
lr =LinearRegression()

lr.fit(x_train,y_train)

predict_lr = lr.predict(x_test)

print('real value y_test[1]:'+str(y_test[1])+'  predict:'+str(lr.predict(x_test.iloc[[1],:])))

print('scort:',lr.score(x_test,y_test))

print('r2 score:',r2_score(y_test,predict_lr))
rfr = RandomForestRegressor()

rfr.fit(x_train,y_train)

predict_rfr = rfr.predict(x_test)

print('real value y_test[1]:'+str(y_test[1])+'  predict:'+str(rfr.predict(x_test.iloc[[1],:])))

print('scort:',rfr.score(x_test,y_test))

print('r2 score:',r2_score(y_test,predict_rfr))
dtr = DecisionTreeRegressor()

dtr.fit(x_train,y_train)

predict_dtr = dtr.predict(x_test)

print('real value y_test[1]:'+str(y_test[1])+'  predict:'+str(dtr.predict(x_test.iloc[[1],:])))

print('scort:',dtr.score(x_test,y_test))

print('r2 score:',r2_score(y_test,predict_dtr))
y = np.array([r2_score(y_test,predict_lr),r2_score(y_test,predict_rfr),r2_score(y_test,predict_dtr)])

x = ['Linear','RandomForest','DecisionTree']

plt.bar(x,y)

plt.xlabel('Regressor')

plt.ylabel('r2 score')