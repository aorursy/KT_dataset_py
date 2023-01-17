import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as st

import datetime as dt

import seaborn as sns
df_auto = pd.read_csv('../input/autos.csv', encoding='latin-1')
analysisCol = ['brand', 'model', 'vehicleType', 'yearOfRegistration', 'monthOfRegistration', 'kilometer', 'powerPS',

               'fuelType', 'gearbox', 'abtest', 'notRepairedDamage', 'seller', 'offerType', 'dateCreated', 

               'lastSeen', 'price']
df_ncar = df_auto[analysisCol]
#Drop rows where Year of registration is >=1950 and <=2016

df_ncar=df_ncar.drop(df_ncar[df_ncar["yearOfRegistration"]<1950].index)

df_ncar=df_ncar.drop(df_ncar[df_ncar["yearOfRegistration"]>2016].index)
#Drop rows with Power >650

df_ncar=df_ncar.drop(df_ncar[df_ncar['powerPS']>650].index)

df_ncar=df_ncar.drop(df_ncar[df_ncar['powerPS']<40].index)
#Drop rows where cars are priced below 1000 and above 35000

df_ncar=df_ncar.drop(df_ncar[df_ncar['price']<1000].index)

df_ncar=df_ncar.drop(df_ncar[df_ncar['price']>35000].index)
#Drop rows where monthOfRegistration is 0

df_ncar = df_ncar.drop(df_ncar[df_ncar['monthOfRegistration']==0].index)
#Convert date columns to date datatype

df_ncar["lastSeen"]=pd.to_datetime(df_ncar["lastSeen"])

df_ncar["dateCreated"]=pd.to_datetime(df_ncar["dateCreated"])

#Extracting the date from datetime

df_ncar["lastSeenDate"]=df_ncar["lastSeen"].dt.date

df_ncar["dateCreatedDate"]=df_ncar["dateCreated"].dt.date
#Create column for date of registration of the car

df_ncar["registerMonthYear"] = df_ncar["yearOfRegistration"].astype(str).str.cat(df_ncar["monthOfRegistration"].astype(str), sep='-')

#Add the day of the month as 01 to the above column since date of the registration wasn't supplied (assuming this wouldn't seriously affect results)

df_ncar["registerDate"]=df_ncar["registerMonthYear"].astype(str)+'-01'

df_ncar["registerDate"]=pd.to_datetime(df_ncar["registerDate"])

df_ncar['registerDate']=df_ncar['registerDate'].dt.date
#Calculate age of the car, in terms of days, at the time the listing was last seen

df_ncar["carAge"]=df_ncar["lastSeenDate"]-df_ncar["registerDate"]
#Remove the word 'days' from the carAge column

       # first convert listAge and carAge to string data type

df_ncar["carAge"]=df_ncar["carAge"].astype(str)

df_ncar["carAge"]=df_ncar.carAge.str.split(n=1).str.get(0)

df_ncar["carAge"]=df_ncar["carAge"].astype(str).astype(int)
df_ncar=df_ncar.drop(df_ncar[df_ncar['carAge']<=0].index)
#Plot distribution of cars

%matplotlib inline

plt.figure(figsize=(13,7))

plt.title('Brand Distribution')

g = sns.countplot(df_ncar['brand'])

rot = g.set_xticklabels(g.get_xticklabels(), rotation=90)
#Plot distribution of car models for a particular brand - I'm taking Volkswagen

%matplotlib inline

vw_cars = df_ncar[df_ncar['brand']=='volkswagen'] #Select data by a brand

plt.figure(figsize=(10,5))

plt.title('VW Car Model Distribution')

bm = sns.countplot(vw_cars['model'])

rot_bm=bm.set_xticklabels(bm.get_xticklabels(), rotation=90)
#Standardize Power PS and Kilometer variables

df_ncar["s_powerPS"]=((df_ncar.powerPS-df_ncar.powerPS.mean())/df_ncar.powerPS.std())

df_ncar["s_kilometer"]=((df_ncar.kilometer-df_ncar.kilometer.mean())/df_ncar.kilometer.std())
#Removing all the categorical columns and columns that have standardized or have been used to create new columns

df_newCar = df_ncar.drop(['vehicleType','yearOfRegistration','monthOfRegistration','kilometer','powerPS',

                          'fuelType','gearbox','abtest','notRepairedDamage','seller','offerType','dateCreated','lastSeen',

                         'lastSeenDate','dateCreatedDate','registerMonthYear','registerDate'], axis=1)
#Building model for just one model - here I chose to build for VW Golf

df_newCar_volkswagenGolf=df_newCar

df_newCar_volkswagenGolf['brand'].replace('volkswagen',1, inplace=True)

df_newCar_volkswagenGolf['model'].replace('golf',1, inplace=True)

df_newCar_volkswagenGolf=df_newCar_volkswagenGolf.drop(df_newCar_volkswagenGolf[df_newCar_volkswagenGolf['brand']!=1].index)

df_newCar_volkswagenGolf=df_newCar_volkswagenGolf.drop(df_newCar_volkswagenGolf[df_newCar_volkswagenGolf['model']!=1].index)
#Distribution of carAge, s_kilometer, and s_powerPS variables

st.probplot(df_newCar_volkswagenGolf.carAge, plot=plt)
t_carAge=df_newCar_volkswagenGolf.carAge

df_newCar_volkswagenGolf['t_carAge'] = np.power(t_carAge,0.5)

st.probplot(df_newCar_volkswagenGolf['t_carAge'], plot=plt)
st.probplot(df_newCar_volkswagenGolf.s_powerPS, plot=plt)
t_powerPS=df_newCar_volkswagenGolf.s_powerPS.add(2)

df_newCar_volkswagenGolf['t_powerPS']=np.power(t_powerPS,-0.4)

st.probplot(df_newCar_volkswagenGolf['t_powerPS'], plot=plt)
#There is interaction between Power and Kilometer i.e. as the number of kilometers increase the power decreases.

df_newCar_volkswagenGolf['KM_Power']=df_newCar_volkswagenGolf['t_powerPS']*df_newCar_volkswagenGolf['s_kilometer']
#Keeping only the variables that will be used in building the model

vwGolf_columnsToKeep = ['t_carAge','s_kilometer','t_powerPS','KM_Power','price']

df_newCar_volkswagenGolf=df_newCar_volkswagenGolf[vwGolf_columnsToKeep]
#Create Sample dataset from the above created dataset

df_newCar_Sample = df_newCar_volkswagenGolf.sample(frac=0.25, random_state=181)
#Create dataset which only contains predictor/independent variables

x_sample = df_newCar_Sample.drop('price', axis=1)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
#Create the linear regression model

lm.fit(x_sample,df_newCar_Sample.price)
print("Estimated intercept coefficient: ", lm.intercept_)

print("Number of coefficients: ", len(lm.coef_))
#create Test dataset with rows that were not part of the training dataset

df_newCar_test=df_newCar_volkswagenGolf.drop(df_newCar_Sample.index)

x_test = df_newCar_test.drop('price',axis=1)
import sklearn as sk

sk.metrics.r2_score(df_newCar_test.price,lm.predict(x_test))
plt.scatter(df_newCar_test.price,lm.predict(x_test))

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted Prices: $\hat{Y}_i$")

plt.title("Prices vs Predicted Prices: $Y_i$ v $\hat{Y}_i$")