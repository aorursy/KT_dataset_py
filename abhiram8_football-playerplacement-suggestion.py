import pandas as pd

import numpy as np

url="https://datafaculty.s3.us-east-2.amazonaws.com/Indore/song_football-class13.csv"
data=pd.read_csv(url,encoding="latin1")
data.head(2)
###### Performance of football players

### Find a replacement for the player. 

### Name is Alexandre Song, 
data.shape[0]
#### How many clusters should I make?

## We do have a context most of the times about the number of cluster
## 10 to 20 players, 30 cluster
480/20
## Agglomerative Clustering
data_num=data.drop(['Player Id','Last_Name','First_Name'],axis=1)
data_num.head(2)
data_num.isnull().sum()
data_num.describe()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_scaled=scaler.fit_transform(data_num)
data_scaled
from sklearn import cluster
mod1=cluster.AgglomerativeClustering(n_clusters=30)
mod1=mod1.fit(data_scaled)
data['labels_1']=mod1.labels_
data.head(2)
data[data['Last_Name']=='Song']
data[data['labels_1']==5].to_csv("final_candidates.csv",index=False)
url1="https://datafaculty.s3.us-east-2.amazonaws.com/Indore/train.csv"

url2="https://datafaculty.s3.us-east-2.amazonaws.com/Indore/store.csv"
train=pd.read_csv(url1)
store=pd.read_csv(url2)
train.head(2)
store.head(2)
store.shape
train.shape
train['Store'].unique()
### ML based regressors we can build ts models with predictors
store.head(2)
store.shape
train.head(2)
train.shape
data=pd.merge(train,store,on='Store',how="left")
data.shape
data.head(2)
###### Time Information, Store level information <===>

### Day,month,year

data['Date']=pd.to_datetime(data['Date'])
data['Month']=data['Date'].dt.month
data['Year']=data['Date'].dt.year
data.isnull().sum()
data.head(2)
len(data['StoreType'].unique())
len(data['Assortment'].unique())
data['StateHoliday'].unique()
data['SchoolHoliday'].unique()
data['PromoInterval'].unique()
### Which predictors we would want to use.

### Time-information- Day,Month,Year

### Store specific variables -
### Replace the missing values with a default number

### Replace the missing values with a string, "missing", 
data['CompetitionDistance'].head(2)
data['CompetitionOpenSinceMonth'].unique()
#data_cat=data[['StoreType','Assortment','StateHoliday']]
data['CompetitionOpenSinceYear'].unique()
data['CompetitionOpenSinceMonth']=data['CompetitionOpenSinceMonth'].fillna("missing")
data['CompetitionOpenSinceMonth']=data['CompetitionOpenSinceMonth'].map(lambda x: str(x))
data['CompetitionOpenSinceMonth'].unique()
data['CompetitionOpenSinceYear'].fillna('missing')

data['CompetitionOpenSinceYear']=data['CompetitionOpenSinceYear'].map(lambda x: str(x))
data['Promo2SinceWeek'].unique()
data['Promo2SinceYear'].unique()
data['Promo2SinceWeek']=data['Promo2SinceWeek'].fillna("missing")

data['Promo2SinceWeek']=data['Promo2SinceWeek'].map(lambda x : str(x))

data['Promo2SinceYear']=data['Promo2SinceYear'].fillna("missing")

data['Promo2SinceYear']=data['Promo2SinceYear'].map(lambda x: str(x))
data.dtypes
data.isnull().sum()
data['CompetitionDistance']=data['CompetitionDistance'].fillna(data['CompetitionDistance'].mean())
data.isnull().sum()
### How would I treat categorical columns
2642/data.shape[0]
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()

data['CompetitionOpenSinceMonth']=enc.fit_transform(data['CompetitionOpenSinceMonth'])
data.head(2)
enc=LabelEncoder()

data['CompetitionOpenSinceYear']=enc.fit_transform(data['CompetitionOpenSinceYear'])
enc=LabelEncoder()

data['Promo2SinceWeek']=enc.fit_transform(data['Promo2SinceWeek'])
enc=LabelEncoder()

data['Promo2SinceYear']=enc.fit_transform(data['Promo2SinceYear'])
data.dtypes
data.head(2)
data['StateHoliday'].unique()
enc=LabelEncoder()

data['StateHoliday']=data['StateHoliday'].map(lambda x: str(x))

data['StateHoliday']=enc.fit_transform(data['StateHoliday'])
enc=LabelEncoder()

data['StoreType']=enc.fit_transform(data['StoreType'])
enc=LabelEncoder()

data['Assortment']=enc.fit_transform(data['Assortment'])
data.isnull().sum()
features=['Store','DayOfWeek','Open','Promo','StateHoliday','SchoolHoliday','StoreType',\

         'Assortment','CompetitionDistance','CompetitionOpenSinceMonth','CompetitionOpenSinceYear',

         'Promo2','Promo2SinceWeek','Promo2SinceYear','Month','Year']
X=data[features]

y=data['Sales']
#### Split data on train and test

import sklearn.model_selection as model_selection
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.20,random_state=42)
import sklearn.ensemble as ensemble
reg=ensemble.RandomForestRegressor(n_estimators=100,min_samples_leaf=100,n_jobs=-1)
reg=reg.fit(X_train,y_train)
###X_train.isnull().sum()
preds=reg.predict(X_test)
X_test['preds']=preds

X_test['actuals']=y_test
X_test=X_test.sort_values(['Year','Month','DayOfWeek'])
X_test=X_test.reset_index()
X_test
X_test_store1=X_test.query("Store==1").reset_index()
X_test_store1[X_test_store1['actuals']==0]
import plotly.express as px
fig=px.line(X_test_store1,x=X_test_store1.index,y="preds")

fig=fig.add_scatter(x=X_test_store1.index,y=X_test_store1['actuals'],name="actual")

fig