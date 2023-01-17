import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

import seaborn as sns

import plotly as plotly                # Interactive Graphing Library for Python

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot, plot

init_notebook_mode(connected=True)

import sklearn

from sklearn import preprocessing

from sklearn.linear_model import Ridge

from sklearn import metrics

from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
airbnb = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

airbnb.head()
airbnb.isnull().sum()
airbnb.drop(["id","host_name","last_review"],axis=1,inplace=True)

airbnb.head()
airbnb.fillna({"reviews_per_month":0},inplace=True)

airbnb.reviews_per_month.isnull().sum()
fig = plt.subplots(figsize = (12,5))

sns.countplot(x = "room_type"  ,hue = "neighbourhood_group",data=airbnb)
a  = airbnb.groupby("neighbourhood_group")["neighbourhood"].value_counts()

b = a.index.levels[0]

a14 = []

a16 = []

for i in range(len(b)):

    df_level = a.loc[[b[i],"neighbourhood"]]

    df_level_ch = pd.DataFrame(df_level)

    for j in range(1):

        a13 = df_level_ch.iloc[j].name

        b1 = df_level_ch.values[j][0]

        print(a13,b1)

        a16.append(b1)

        a14.append(a13)

a15 = pd.DataFrame(a14)

a16 = pd.DataFrame(a16)

a17 = pd.concat([a15,a16],axis=1)

a17.columns=["地区","区域","数量"]

a17.plot.bar(x="区域",y="数量")
airbnb.price.mean()  
plt.figure(figsize = (16,8))

su_5 = airbnb[airbnb.price < 700]

sns.violinplot(data=su_5,x="neighbourhood_group",y="price")
le=preprocessing.LabelEncoder()   

le.fit(airbnb["neighbourhood_group"]) #标签化

airbnb["neighbourhood_group"] = le.transform(airbnb["neighbourhood_group"])



le=preprocessing.LabelEncoder()

le.fit(airbnb["neighbourhood"])

airbnb["neighbourhood"] = le.transform(airbnb["neighbourhood"])

airbnb["neighbourhood"]



le=preprocessing.LabelEncoder()

le.fit(airbnb['room_type'])

airbnb['room_type'] = le.transform(airbnb['room_type'])



airbnb.sort_values(by="price",ascending=True,inplace=True)



airbnb.head()
lm = LinearRegression()

X = airbnb[['neighbourhood_group','neighbourhood','latitude','longitude','room_type',

            'minimum_nights','number_of_reviews','reviews_per_month',

            'calculated_host_listings_count','availability_365']]

y = airbnb["price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)

lm.fit(X_train,y_train)
score = lm.score(X_test,y_test)#预测准确率

score
yhat = lm.predict(X_test)

plt.figure(figsize=(16,8))

sns.regplot(yhat,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.grid(False)

plt.show()
GBoost = GradientBoostingRegressor(n_estimators=1000,learning_rate=0.01)

GBoost.fit(X_train,y_train)

score2 = GBoost.score(X_test,y_test)

score2
yhat2 = GBoost.predict(X_test)   #预测值

plt.figure(figsize=(16,8))

sns.regplot(yhat2,y_test)

plt.xlabel('Predictions')

plt.ylabel('Actual')

plt.title("Linear Model Predictions")

plt.show()