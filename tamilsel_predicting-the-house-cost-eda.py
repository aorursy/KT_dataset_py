import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Loading Data



df = pd.read_csv("/kaggle/input/brasilian-houses-to-rent/houses_to_rent.csv")

df.head()
df = df.drop(["Unnamed: 0","floor"], axis = 1)

df.head()
#Checking for missing values



df.isnull().sum()
def remove_dollor(x):

    a =  x[2:] #removes first two chr

    result = ""

    for i in a:

        if i.isdigit() is True:

            result = result + i

    return result #returns only digits (excludes special character)
df["hoa"] = pd.to_numeric(df["hoa"].apply(remove_dollor), errors= "ignore")

df["rent amount"] = pd.to_numeric(df["rent amount"].apply(remove_dollor), errors= "ignore")

df["property tax"] = pd.to_numeric(df["property tax"].apply(remove_dollor), errors= "ignore")

df["fire insurance"] = pd.to_numeric(df["fire insurance"].apply(remove_dollor), errors= "ignore")

df["total"] = pd.to_numeric(df["total"].apply(remove_dollor), errors= "ignore")
df.dtypes
df.head()
plt.figure(figsize = (5,5))

sns.boxplot(df["total"])
#Since we found many outlliers, let's try to remove.



q1 = df["total"].quantile(0.25)

q3 = df["total"].quantile(0.75)



IQR = q3 - q1

IF = q1 - (1.5 * IQR)

OF = q3 + (1.5 * IQR)
data = df[~((df["total"] < IF) | (df["total"] > OF))]

data.shape
data.head()
print("Before Outlier Removal")

print("No. of rows : ", df.shape[0])

print("No. of columns : ", df.shape[1])

print("=======================")

print("After Outlier Removal")

print("No. of rows : ", data.shape[0])

print("No. of columns : ", data.shape[1])
#Before Removing Outliers

plt.figure(figsize = (7,7))

sns.set(style = "whitegrid")

f = sns.barplot(x = "rooms", y = "total", data = df)

f.set_title("Before Removing Outliers")

f.set_xlabel("No. of Rooms")

f.set_ylabel("Total Cost")
#After Removing Outliers

plt.figure(figsize = (7,7))

sns.set(style = "whitegrid")

f = sns.barplot(x = "rooms", y = "total", data = data)

f.set_title("After Removing Outliers")

f.set_xlabel("No. of Rooms")

f.set_ylabel("Total Cost")
df.columns
columns = ["city","rooms","bathroom","parking spaces", "animal", "furniture"]

plt.figure(figsize = (30,30))

for i,var in enumerate(columns,1):

    plt.subplot(2,4,i)

    f = sns.barplot(x = data[var], y = data["total"])

    f.set_xlabel(var.upper())

    f.set_ylabel("Total Cost")
plt.figure(figsize=(12,12))

sns.heatmap(data.corr(), annot=True)
plt.figure(figsize = (7,7))

sns.set(style = "whitegrid")

f = sns.distplot(data["total"])
data["animal"].value_counts()
animal_dict = {"acept": 1,"not acept":0}

data["animal_en"] = data["animal"].map(animal_dict)

data.head()
data["furniture"].value_counts()
furniture_dict = {"furnished":1, "not furnished":0}

data["furniture_en"] = data["furniture"].map(furniture_dict)
data = data.drop(["animal","furniture"], axis = 1)
data.head()
data["hoa"] = df["hoa"].fillna(data["hoa"].median())

data["property tax"] = data["property tax"].fillna(data["property tax"].median())
x = data.drop("total", axis = 1)

y = data["total"]
#Spliting train and test



from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
#Standardization train data

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

std.fit(x_train)

std_data = std.transform(x_train)
#Box-Cox transformation for train response variable

from scipy import stats

bx, lam = stats.boxcox(y_train)

y_total = bx
from sklearn.ensemble import RandomForestRegressor

#Training the model

rf = RandomForestRegressor(n_estimators = 1500)

rf.fit(std_data, y_total)
#Standardization test data

from sklearn.preprocessing import StandardScaler

std = StandardScaler()

std.fit(x_test)

std_test = std.transform(x_test)
#Prediction

pred = rf.predict(std_test)
#Box-Cox transformation for Test Response variable

bx, lam = stats.boxcox(y_test)

y_total_test = bx
def rmse_test(ytest, pred,xtest):

    err = ytest - pred

    mse = sum(err**2)/(xtest.shape[0]-xtest.shape[1]-1)

    rmse = np.sqrt(mse)

    print("RMSE OF TEST DATA IS : ", rmse)
rmse_test(y_total_test,pred,x_test)