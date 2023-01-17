import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
ny=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

ny
ny.head()
ny.info()
num_cols=ny.select_dtypes(exclude="object").columns

num_cols
cat_cols=ny.select_dtypes(include="object").columns

cat_cols
ny.dtypes
ny.isnull().sum()
ny=ny.drop(["id","host_name","last_review"],axis=1)

ny.head()
ny=ny.fillna({"reviews_per_month":0})
ny.isnull().sum()
ny.duplicated().sum()
ny.neighbourhood_group.value_counts().plot(kind="bar")

plt.title("Share of neighbourhood")

plt.xlabel("neighbourhood_group")

plt.ylabel("Count")
ny['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.1,0,0,0],autopct='%1.1f%%',shadow=True)

plt.figure(figsize=(10,6))

sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = ny)

plt.title("Room types occupied by the neighbourhood_group")

plt.show()
sns.barplot(data=ny,x="neighbourhood_group",y="price")

plt.show()
ny["price"].describe()
sns.heatmap(ny.corr(),annot=True,cmap="coolwarm")

plt.show()
f,ax = plt.subplots(figsize=(16,8))

ax = sns.scatterplot(y=ny.latitude,x=ny.longitude,hue=ny.neighbourhood_group,palette="coolwarm")

plt.show()
plt.figure(figsize=(10,6))

ny1=ny[ny.price<500]



ny1.plot(kind='scatter', x='longitude',y='latitude',label='availability_365',c='price',cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,figsize=(10,10))



plt.show()
sns.stripplot(data=ny,x='room_type',y='price',jitter=True)



plt.show()
plt.figure(figsize=(10,6))

ny['number_of_reviews'].plot(kind='hist')

plt.xlabel("Price")

plt.show()
f,ax = plt.subplots(figsize=(25,5))

ax=sns.stripplot(data=ny,x='minimum_nights',y='price',jitter=True)

plt.show()
ny1=ny.sort_values(by=['number_of_reviews'],ascending=False).head(100)

ny1.head()


import statsmodels.api as sm

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

from sklearn import preprocessing

from sklearn.feature_selection import RFE



import warnings 

warnings.filterwarnings('ignore')
ny["name"] = pd.get_dummies(ny['name'])

ny["neighbourhood_group"]= pd.get_dummies(ny['neighbourhood_group'])

ny["neighbourhood"]= pd.get_dummies(ny['neighbourhood'])

ny["room_type"]= pd.get_dummies(ny['room_type'])



ny
X=ny.drop('price',axis=1)

y=ny['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
lr=LinearRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

r2_score(y_test,y_pred)
rfe = RFE(lr, 7)

#Transforming data using RFE

X_rfe = rfe.fit_transform(X,y)  

#Fitting the data to model

lr.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)
#no of features

nof_list=np.arange(1,10)            

high_score=0

#Variable to store the optimum features

nof=0           

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

    lr = LinearRegression()

    rfe = RFE(lr,nof_list[n])

    X_train_rfe = rfe.fit_transform(X_train,y_train)

    X_test_rfe = rfe.transform(X_test)

    lr.fit(X_train_rfe,y_train)

    score = lr.score(X_test_rfe,y_test)

    score_list.append(score)

    if(score>high_score):

        high_score = score

        nof = nof_list[n]

print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))
xc=sm.add_constant(X)

lm=sm.OLS(y,xc).fit()

lm.summary()
X=X.drop("minimum_nights",axis=1)
xc=sm.add_constant(X)

lm=sm.OLS(y,xc).fit()

lm.summary()
plt.figure(figsize=(16,8))

sns.regplot(y_test,y_pred)

plt.xlabel('Actual')

plt.ylabel('Prediction')

plt.title("SLR Model")

plt.grid(False)

plt.show()