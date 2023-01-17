import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection # for splitting the data into training and testing data
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/autos.csv" ,encoding = "ISO-8859-1" )
print(train_data.shape)
train_data.head()
train_data.describe()
train_data.isnull().sum()
train_data["seller"].value_counts()
del train_data["seller"]
train_data["offerType"].value_counts()
del train_data["offerType"]
train_data.head(3) 
train_data["nrOfPictures"].value_counts()
del train_data["nrOfPictures"]
train_data["abtest"].value_counts()
train_data[ ["dateCrawled","dateCreated","lastSeen"] ].head()
train_data = train_data.drop(["dateCrawled","dateCreated","lastSeen"] , axis=1 )
train_data.head()
train_data["name"].head()
del train_data["name"]
train_data.isnull().sum()
train_data["gearbox"].value_counts()
# we can see brand column has no nans
train_data["brand"].isnull().sum()
#so we will use brand column to fill gearbox values
train_data.groupby("brand")["gearbox"].value_counts()
gearbox = train_data["gearbox"].unique()
brand = train_data["brand"].unique()
d = {}

for i in brand :
    m = 0
    for j in gearbox :
        if train_data[(train_data.gearbox == j) & (train_data.brand == i)].shape[0] > m :
            m = train_data[(train_data.gearbox == j) & (train_data.brand == i)].shape[0]
            d[i] = j
        
for i in brand :
    train_data.loc[(train_data.brand == i) & (train_data.gearbox.isnull()) ,"gearbox" ] = d[i]

# no nans in gearbox
train_data["gearbox"].isnull().sum()
train_data["notRepairedDamage"].value_counts()
train_data["notRepairedDamage"].isnull().sum()
train_data["notRepairedDamage"].fillna("nein",inplace = True)
train_data["notRepairedDamage"].isnull().sum()
train_data["fuelType"].value_counts()
train_data["fuelType"].fillna("benzin",inplace = True)
train_data.isnull().sum()
train_data["vehicleType"].value_counts()
# we can fill according to the fuelType values
train_data.groupby("fuelType")["vehicleType"].value_counts()
vehicleType = train_data["vehicleType"].unique()
fuelType = train_data["fuelType"].unique()
print(fuelType)
print(vehicleType)
#remove nan 
vehicleType = np.delete(vehicleType,0)
d = {}
for i in fuelType :
    m = 0
    for j in vehicleType :
        if train_data[(train_data.vehicleType == j) & (train_data.fuelType == i)].shape[0] > m :
            m = train_data[(train_data.vehicleType == j) & (train_data.fuelType == i)].shape[0]
            d[i] = j
for i in fuelType :
    train_data.loc[(train_data.fuelType == i) & (train_data.vehicleType.isnull()) ,"vehicleType" ] = d[i]

train_data["vehicleType"].isnull().sum()
len(train_data["model"].unique())
train_data["model"].unique()[0]
train_data["model"].fillna("golf",inplace =True)
train_data.isnull().sum()
train_data.head()
train_data["postalCode"].head()
del train_data["postalCode"]
from sklearn.preprocessing import LabelEncoder
data = train_data.copy()
data["vehicleType"] =LabelEncoder().fit_transform(data["vehicleType"])
data["fuelType"] =LabelEncoder().fit_transform(data["fuelType"])
data["gearbox"] =LabelEncoder().fit_transform(data["gearbox"])
data["notRepairedDamage"] =LabelEncoder().fit_transform(data["notRepairedDamage"])
data["brand"] =LabelEncoder().fit_transform(data["brand"])
data["model"] =LabelEncoder().fit_transform(data["model"])
data["abtest"] =LabelEncoder().fit_transform(data["abtest"])

# analysis year o registration
data["yearOfRegistration"].describe()
data[data.yearOfRegistration > 2017].shape
data[data.yearOfRegistration < 1950].shape
data = data[(data.yearOfRegistration < 2017)  & (data.yearOfRegistration > 1950)]
# now lets look at the price
data["price"].describe()
data[data.price < 100].shape
data[data.price > 200000].shape
data = data[(data.price > 100) & (data.price < 200000) ]
# lets seperate the output and input
y  = data["price"]
x =  data.drop("price",axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0)
# classifier
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr.score(x_test, y_test)