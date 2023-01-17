import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/zomato.csv", encoding = 'ISO-8859-1')
df.head(10)
to_drop = ["Locality", "Address", "Locality Verbose", "Longitude", "Latitude" , "Switch to order menu"]

df.drop(to_drop, inplace = True, axis =1 )
df['Restaurant ID'].is_unique
df["Country Code1"]=df["Country Code"].apply(str)

df['Country Code']=df['Country Code'].replace({189:'Canada',216:'Tunisia',215:'Philadelphia',214:'Dallas',1:'India',30:'Greece',148:'Equador'})

df['Country Code']=df['Country Code'].replace([208,14,94,191,162,184,166,37],'Others')

df=df.rename(columns={"Country Code":"Country Name"})
df[df["Average Cost for two"]>450000]
df=df[df["Restaurant ID"] != 7402935]

df=df[df["Restaurant ID"] != 7410290]

df=df[df["Restaurant ID"] != 7420899]
df['Has Table booking'] = pd.get_dummies(df["Has Table booking"],drop_first=True)

df['Has Online delivery'] = pd.get_dummies(df["Has Online delivery"],drop_first=True)

df['Is delivering now'] = pd.get_dummies(df["Is delivering now"],drop_first=True)

df['Currency']=df['Currency'].replace({'Dollar($)':'Dollar','Pounds(��)':'Pounds','Brazilian Real(R$)':'Brazilian Real','NewZealand($)':'NewZealand Dollar'})

cus=df["Cuisines"].value_counts()

cuisines = {}

cnt=0

for i in cus.index:

    for j in i.split(", "):

        if j not in cuisines:

            cuisines[j]=cus[cnt]

        else:

            cuisines[j] += cus[cnt]

    cnt += 1

    

cuisines = pd.Series(cuisines).sort_values(ascending=False)
India=df[df.Currency == 'Indian Rupees(Rs.)']

q3_v=India["Votes"].quantile(0.75)

q1_v=India["Votes"].quantile(0.25)

iqr_v=q3_v-q1_v

lowervotes=q1_v-(iqr_v*1.5)

uppervotes=q3_v+(iqr_v*1.5)

uppervotes
India=India[India["Votes"]<244]

q3_avg=India["Average Cost for two"].quantile(0.75)

q1_avg=India["Average Cost for two"].quantile(0.25)

iqr_avg=q3_avg-q1_avg

loweravg=q1_avg-(iqr_avg*1.5)

upperavg=q3_avg+(iqr_avg*1.5)

upperavg
India=India[India["Average Cost for two"]<1050]

X=India.drop(["Restaurant ID","Restaurant Name","Rating text","Country Name","City","Rating color",

           "Cuisines","Currency","Country Code1","Aggregate rating"],axis=1)

y=India["Aggregate rating"]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test  = train_test_split(X, y , test_size = 0.2, random_state = 42)
model = LinearRegression()

model.fit(X_train, y_train)
y_predict =  model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_predict)
from sklearn.tree import DecisionTreeRegressor



modeldt= DecisionTreeRegressor(max_depth=6)

modeldt.fit(X_train,y_train)
y_predictdt=modeldt.predict(X_test)

r2_score(y_test,y_predictdt)
from sklearn.ensemble import GradientBoostingRegressor

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,

      max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)

y_predictdt=est.predict(X_test)

r2_score(y_test,y_predictdt)
import xgboost as xgb

xgb_clf = xgb.XGBRegressor(max_depth=3, n_estimators=5000, learning_rate=0.2,

                            n_jobs=-1)
xgb_clf.fit(X_train, y_train)



y_pred = xgb_clf.predict(X_test)
r2_score(y_test, y_pred) 