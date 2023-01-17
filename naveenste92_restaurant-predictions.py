#importing required packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import string
import os
#reading data
F1=pd.read_csv("../input/chefmozaccepts.csv")
F2=pd.read_csv("../input/chefmozcuisine.csv")
F3=pd.read_csv("../input/chefmozhours4.csv")
F4=pd.read_csv("../input/chefmozparking.csv")
F5=pd.read_csv("../input/usercuisine.csv")
F6=pd.read_csv("../input/userpayment.csv")
F7=pd.read_csv("../input/userprofile.csv")
F8=pd.read_csv("../input/geoplaces2.csv")

T=pd.read_csv("../input/rating_final.csv")
F1.head()
F1.info()
# plot to visualise most accepted payments by Restaurants
F1plt=F1.Rpayment.value_counts().plot.bar(title="Payments Accepted")
F1plt.set_xlabel('payments mode',size=15)
F1plt.set_ylabel('count',size=15)
#creating dummy variables for differen payments.
F1dum = pd.get_dummies(F1,columns=['Rpayment'])
F1dum1 = F1dum.groupby('placeID',as_index=False).sum()
len(F1dum1)
F1dum1.head()
F2.head()
F2.info()
#plot to visualize top cuisines offered by the restaurants
F2plt=F2.Rcuisine.value_counts()[:10].plot.bar(title="Top 10 cuisine")
F2plt.set_xlabel('cuisine',size=15)
F2plt.set_ylabel('count',size=15)
#creating dummy variables for different cuisines.
F2dum = pd.get_dummies(F2,columns=['Rcuisine'])
F2dum1 = F2dum.groupby('placeID',as_index=False).sum()
len(F2dum1)
F2dum1.head()
F3.head()
F3.info()
F4.head()
F4.info()
#plot to visualize available parking place at the Restaurants
F4plt=F4.parking_lot.value_counts().plot.bar(title="parking place")
F4plt.set_xlabel('Available parking',size=15)
F4plt.set_ylabel('count',size=15)
#creating dummy variables for different parking lots.
F4dum = pd.get_dummies(F4,columns=['parking_lot'])
F4dum1 = F4dum.groupby('placeID',as_index=False).sum()
len(F4dum1)
F4dum1.head()
F5.head()
F5.info()
#Top 10 favorite cuisines for the customers
F5plt=F5.Rcuisine.value_counts()[:10].plot.bar(title="Top 10 user cuisine")
F5plt.set_xlabel('user cuisine',size=15)
F5plt.set_ylabel('count',size=15)
#creating dummy variables for differen usercuisines.
F5dum = pd.get_dummies(F5,columns=['Rcuisine'])
F5dum1 = F5dum.groupby('userID',as_index=False).sum()
len(F5dum1)
F5dum1.head()
F6.head()
F6.info()
#top type of payments done by the users
F6plt=F6.Upayment.value_counts().plot.bar(title="User payments")
F6plt.set_xlabel('User payments',size=15)
F6plt.set_ylabel('count',size=15)
#creating dummy variables for different userpayments.
F6dum = pd.get_dummies(F6,columns=['Upayment'])
F6dum1 =F6dum.groupby('userID',as_index=False).sum()
len(F6dum1)
F6dum1.head()
F7.head()
F7.info()
# as data contains unknown value, we are replacinf with Nan.
F7rep=F7.replace('?', np.nan)
#now we are finding missing value cnt n perct for all variables.
mss=F7rep.isnull().sum()
columns = F7rep.columns
percent_missing = F7rep.isnull().sum() * 100 / len(F7rep)
missing_value_F7rep = pd.DataFrame({'missing_cnt':mss,'percent_missing': percent_missing})
missing_value_F7rep
#since the missing value pernt is very low in each variables, we are replacing with mode of that individual column.
for column in F7rep.columns:
    F7rep[column].fillna(F7rep[column].mode()[0], inplace=True)
#plotting for marital status vs smoker n drinklevel.
F7rep.groupby('marital_status')['smoker','drink_level'].nunique().plot.bar(rot=0)
#plot to visualize user's personal info based on birthyear.
F7repplt=F7rep.groupby('birth_year')['interest','personality','religion','activity'].nunique().plot.bar(figsize=(15, 5))
#now performing label encoding to convert char to factors.
F7char=F7rep.select_dtypes(include=['object'])

encoder = LabelEncoder()
F7charLE = F7char.apply(encoder.fit_transform, axis=0)
F7charLE=F7charLE.drop(['userID'],axis=1)
F7charLE[['userID','latitude','longitude','birth_year','weight','height']]=F7rep[['userID','latitude','longitude','birth_year','weight','height']]
F7charLE.head()


F8.head()
F8.info()
#replacing unknown value with Nan.
F8rep=F8.replace('?', np.nan)
#now we are finding missing value cnt n perct for all variables.
mss=F8rep.isnull().sum()
columns = F8rep.columns
percent_missing = F8rep.isnull().sum() * 100 / len(F8rep)
missing_value_F8rep = pd.DataFrame({'missing_cnt':mss,
                                 'percent_missing': percent_missing})
missing_value_F8rep
#dropping columns with more than 50% missing values
F8new=F8rep.drop(['fax','zip','url'],axis=1)
#and replacing remaining colvalues with mode
for column in F8new.columns:
    F8new[column].fillna(F8new[column].mode()[0], inplace=True)
#clean n cnt of city
F8new.city=F8new.city.apply(lambda x: x.lower())
F8new.city=F8new.city.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

F8new.city.value_counts()
#replacing city with unique. 
F8new['city']=F8new['city'].replace(['san luis potos','san luis potosi','slp','san luis potosi '],'san luis potosi' )
F8new['city']=F8new['city'].replace(['victoria','cd victoria','victoria '],'ciudad victoria' )
F8new.city.value_counts()
#clean n cnt of state
F8new.state=F8new.state.apply(lambda x: x.lower())
F8new.state=F8new.state.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

F8new.state.value_counts()
#replacing state with unique.
F8new['state']=F8new['state'].replace(['san luis potos','san luis potosi','slp'],'san luis potosi' )
F8new.state.value_counts()
#clean n cnt of country
F8new.country=F8new.country.apply(lambda x: x.lower())
F8new.country=F8new.country.apply(lambda x:''.join([i for i in x 
                            if i not in string.punctuation]))

F8new.country.value_counts()
#label encoding
F8char=F8new.select_dtypes(include=['object'])
F8charLE = F8char.apply(encoder.fit_transform, axis=0)
F8charLE[['placeID','latitude','longitude']]=F8new[['placeID','latitude','longitude']]
F8charLE.head()
#plot for facilities provided by Restaurants based on city.
F8newplt=F8new.groupby('city')['alcohol','smoking_area','accessibility','price','Rambience','other_services'].nunique().plot.bar(figsize=(15,5))
mapbox_access_token='pk.eyJ1IjoibmF2ZWVuOTIiLCJhIjoiY2pqbWlybTc2MTlmdjNwcGJ2NGt1dDFoOSJ9.z5Jt4XxKvu5voCJZBAenjQ'
mcd=F8rep[F8rep.country =='Mexico']
mcd_lat = mcd.latitude
mcd_lon = mcd.longitude

data = [
    go.Scattermapbox(
        lat=mcd_lat,
        lon=mcd_lon,
        mode='markers',
        marker=dict(
            size=6,
            color='rgb(255, 0, 0)',
            opacity=0.4
        ))]
layout = go.Layout(
    title='Restaurants Locations',
    autosize=True,
    hovermode='closest',
    showlegend=False,
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=23,
            lon=-102
        ),
        pitch=2,
        zoom=4.5,
        style='dark'
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='restaurants')
#merging ratingfile(T) with userprofile(F7)
A=pd.merge(T,F7charLE)
#merging A with userpayments(F6) 
B=pd.merge(A,F6dum1,how='left',on=['userID'])
#merging B with usercuisine(F5)
C=pd.merge(B,F5dum1,how='left',on=['userID'])
#merging C with geoplaces2(F8)
D=pd.merge(C,F8charLE,how='left',on=['placeID'])
#merging D with chefmozparking(F4)
E=pd.merge(D,F4dum1,how='left',on=['placeID'])
#merging E with chefmozcuisine(F2)
F=pd.merge(E,F2dum1,how='left',on=['placeID'])
#merging F with chefmozaccepts(F1)
G=pd.merge(F,F1dum1,how='left',on=['placeID'])
len(G)
G.head()
G.info()
print('No of columns',G.shape[1])
print('No of rows',G.shape[0])
#check for Null values
G.isnull().values.any()
#finding percentage of null values across columns
columns = G.columns
percent_missing = G.isnull().sum() * 100 / len(G)
missing_value_G = pd.DataFrame({'percent_missing': percent_missing})
missing_value_G
#replacing missing values with zero and check.
G=G.fillna(0)
G.isnull().values.any()
#for modelling purpose we are label encoding userID.
G['userID']=encoder.fit_transform(G['userID'])
#packages for modelling
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
# splitting train and test data as 75/25.
X=G.drop(['placeID','rating','food_rating','service_rating'],axis=1)
y=G['rating']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
#model building.
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
#predicting on test data.
predictions1 =  logmodel.predict(X_test)
print("confusion matrix")
print(confusion_matrix(y_test,predictions1))
print("Accuracy_score")
print(accuracy_score(y_test, predictions1))
print("classification_report")
print(classification_report(y_test,predictions1))
#kappa score.
cohen_kappa_score(y_test, predictions1)
#model building.
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
#predicting on test data.
predictions2 =  clf.predict(X_test)
print("confusion matrix")
print(confusion_matrix(y_test,predictions2))
print("Accuracy_score")
print(accuracy_score(y_test, predictions2))
print("classification_report")
print(classification_report(y_test,predictions2))
#kappa score.
cohen_kappa_score(y_test, predictions2)
#model building.
Rndclf = RandomForestClassifier(max_depth=2, random_state=0) 
Rndclf.fit(X_train,y_train)
#predicting on test data.
predictions3 = Rndclf.predict(X_test)
print("confusion matrix")
print(confusion_matrix(y_test,predictions3))
print("Accuracy_score")
print(accuracy_score(y_test, predictions3))
print("classification_report")
print(classification_report(y_test,predictions3))
#kappa score.
cohen_kappa_score(y_test, predictions3)
#model building.
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
#predicting on test data.
predictions4 = xgb.predict(X_test)
print("confusion matrix")
print(confusion_matrix(y_test,predictions4))
print("Accuracy_score")
print(accuracy_score(y_test, predictions4))
print("classification_report")
print(classification_report(y_test,predictions4))
#kappa score.
cohen_kappa_score(y_test, predictions4)