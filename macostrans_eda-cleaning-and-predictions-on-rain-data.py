#Lets Start

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

print(os.listdir("../input"))
data_raw = pd.read_csv(r"../input/weatherAUS.csv")

data_raw.head()
data_raw.info()
import missingno as msno

msno.matrix(data_raw)
data_raw2 = data_raw.drop(axis=1,columns=["Evaporation", "Sunshine","Cloud9am","Cloud3pm"])
data_raw2['Date'] = pd.to_datetime(arg=data_raw2['Date'])
data_raw2["Month"] = data_raw2["Date"].dt.month

data_raw2["Year"] = data_raw2["Date"].dt.year

data_raw2["Day"] = data_raw2["Date"].dt.day
data_mod1 = data_raw2.dropna(how='any',axis=0)
fig, ax = plt.subplots(figsize=[12,12])

sns.heatmap(data_mod1.corr(),fmt=".2f",annot=True)
data_mod1.describe()
#Multiplotting

plt.figure(1,figsize=[14,12])



#Subplot 1

plt.subplot(2,2,1)

sns.boxplot(data=data_mod1,y="Humidity9am",x="RainTomorrow")

plt.title("Humidity in the morning vs RainingTomorrow")

plt.xlabel("Has it Rained")

plt.ylabel("humidity at 9am")



#Subplot 2

plt.subplot(2,2,2)

sns.boxplot(data=data_mod1,y="Pressure9am",x="RainTomorrow")

plt.title("Morning Pressure vs RainingTomorrow")

plt.xlabel("Has it Rained")

plt.ylabel("Pressure at 9am")



#Subplot 3

plt.subplot(2,2,3)

sns.boxplot(data=data_mod1,y="Humidity3pm",x="RainTomorrow")

plt.title("Humidity in the evening vs RainingTomorrow")

plt.xlabel("Has it Rained")

plt.ylabel("humidity at 3pm")



#Subplot 4

plt.subplot(2,2,4)

sns.boxplot(data=data_mod1,y="Pressure3pm",x="RainTomorrow")

plt.title("Evening Pressure vs RainingTomorrow")

plt.xlabel("Has it Rained")

plt.ylabel("Pressure at 3pm")
#Multiplotting

plt.figure(1,figsize=[14,12])



#Subplot 1

plt.subplot(2,2,1)

sns.boxplot(data=data_mod1,y="MinTemp",x="Month")

plt.title("Min Temparature Monthwise")

plt.xlabel("Month")

plt.ylabel("Temparature")

plt.ylim([-10,50])



#Subplot 2

plt.subplot(2,2,2)

sns.boxplot(data=data_mod1,y="MaxTemp",x="Month")

plt.title("Max Temp Monthwise")

plt.xlabel("Month")

plt.ylabel("Temparature")

plt.ylim([-10,50])



#Subplot 3

plt.subplot(2,2,3)

sns.boxplot(data=data_mod1,y="Humidity9am",x="Month")

plt.title("Morning Humidity Monthwise")

plt.xlabel("Month")

plt.ylabel("Morning Humidity")



#Subplot 4

plt.subplot(2,2,4)

sns.boxplot(data=data_mod1,y="Humidity3pm",x="Month")

plt.title("Evening Humidity Monthwise")

plt.xlabel("Month")

plt.ylabel("Evening Humidity")
data_mod1["RainToday"] = data_mod1["RainToday"].map({"No":0,"Yes":1})

data_mod1["RainTomorrow"] = data_mod1["RainTomorrow"].map({"No":0,"Yes":1})

data_mod1.head()
WindGustDir_dummies = pd.get_dummies(data_mod1["WindGustDir"],drop_first=True,prefix="WindGustDir")

WindDir9am_dummies = pd.get_dummies(data_mod1["WindDir9am"],drop_first=True,prefix="WindDir9am")

WindDir3pm_dummies = pd.get_dummies(data_mod1["WindDir3pm"],drop_first=True,prefix="WindDir3pm")

Location_dummies = pd.get_dummies(data_mod1["Location"],drop_first=True,prefix="Country")

data_mod2=pd.concat([data_mod1,Location_dummies, WindDir3pm_dummies,WindDir9am_dummies,WindGustDir_dummies],axis=1)

data_mod2.head()
data_mod2.drop(["Location","Date", "WindGustDir","WindDir9am","WindDir3pm"],inplace=True,axis=1)

cols = list(data_mod2.columns.values) #Make a list of all of the columns in the df

cols.pop(cols.index('RainTomorrow')) #Remove b from list

data_mod2 = data_mod2[cols+["RainTomorrow"]]

data_mod2.head()
X = data_mod2.iloc[:, :-1].values

y = data_mod2.loc[:,"RainTomorrow"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

# Predicting the Test set results

y_pred = classifier.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# print(cm)

plt.figure(figsize = (8,8))

sns.heatmap(cm,fmt="d",annot=True,xticklabels=["Not Raining","Raining"],yticklabels=["Not Raining","Raining"],cbar=False)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actuals")

plt.show()
from sklearn.metrics import accuracy_score

print("Accuracy of the model: "+ str(accuracy_score(y_test, y_pred)*100) + "%")