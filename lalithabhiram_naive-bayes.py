import numpy as np

import pandas as pd
import pandas as pd

Abhi = pd.read_csv("../input/User_Data.csv")
Abhi.head()
X = Abhi.iloc[:,[2]].values

y = Abhi.iloc[:, 4].values
print("The Column of Age is",X)
print("The Column of Purchasement is",y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
print(X_train)
print(X_test)
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB() 

gnb=gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
print("The Predicted values in the model are: \n",y_pred)