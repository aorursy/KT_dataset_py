import numpy as np 

import pandas as pd

import os

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print(os.listdir("../input"))
data=pd.read_csv("../input/online_shoppers_intention.csv")

df=data.copy()

df.head()
df.dropna()

df.info()
df.describe().T
import seaborn as sns

f,ax=plt.subplots(2,1,figsize=(10,10))

sns.countplot(df["VisitorType"],palette="hls",ax=ax[0]);

sns.countplot(df["VisitorType"],hue=df["Revenue"],palette="hls",ax=ax[1]);
f,ax=plt.subplots(2,1,figsize=(8,8))

sns.countplot(df["Weekend"],palette="coolwarm",ax=ax[0]);

sns.countplot(df["Weekend"],hue=df["Revenue"],palette="Set1",ax=ax[1]);
f,ax=plt.subplots(2,1,figsize=(12,10))

sns.countplot(df["Month"],palette="cubehelix",ax=ax[0]);

sns.countplot(df["Month"],hue=df["Revenue"],palette="Blues",ax=ax[1]);
df["BounceRates"]=df["BounceRates"].fillna(0)

plt.figure(figsize=(15,8))

sns.distplot(df["BounceRates"],color="green");
plt.figure(figsize=(20,10))

sns.heatmap(df.corr(),annot=True,linewidths=.05);
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

#df["VisitorType"].unique #Returning_Visitor ve New Visitor olarak 2 çeşit bunları 0 ve 1 olarak değiştiriyoruz.
visitor_type=le.fit_transform(df["VisitorType"].values)

print(le.classes_)

visitor_type
weekend=le.fit_transform(df["Weekend"].values)

print(le.classes_)

weekend
month=le.fit_transform(df["Month"].values)

print(le.classes_)

month
df["BounceRates"]=df["BounceRates"].fillna(0)
X=pd.concat([pd.DataFrame(visitor_type),pd.DataFrame(weekend),pd.DataFrame(month),df["BounceRates"],df["SpecialDay"]],axis=1)

X.columns=["visitor_type","weekend","month","bouncerates","specialday"]

X.head()
y=le.fit_transform(df["Revenue"])

y=pd.DataFrame(y,columns=["revenue"])

y.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=31)
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

model_nb=nb.fit(X_train,y_train)

model_nb
y_pred=model_nb.predict(X_test)

accuracy_score(y_test,y_pred)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()



scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)

X_test_scaled=scaler.transform(X_test)

mlpc=MLPClassifier().fit(X_train_scaled,y_train)

mlpc
y_pred=mlpc.predict(X_test_scaled)

accuracy_score(y_test,y_pred)