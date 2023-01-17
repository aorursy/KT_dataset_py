1

import pandas as pd
data=pd.read_csv('../input/flight-route-database/routes.csv')

nan=data[data.isnull().any(axis=1)].head()

nan.iloc[0:3]
data2=data.fillna(0)

data2.head()
2

import matplotlib.pyplot as plt
team=['CSK','KKR','DC','MI']

score=[149,218,188,143]

plt.bar(team,score,color=['gold','purple','gold','gold'])

plt.title('IPL TEAM SCORE GRAPH')

plt.xlabel('TEAMS')

plt.ylabel('SCORE')
3

import numpy as np
a1=np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])

a2=np.array([2,5,6,10,80])

temp=[m for m, val in enumerate(a1) if val in set(a2)]

new_arr=np.delete(a1,temp)

print("ARRAY 1:",a1)

print("ARRAY 2:",a2)

print("NEW ARRAY:",new_arr)

print("ARRAY 2:",a2)
4

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



train = pd.read_csv("../input/titanic/train_data.csv")





X = train.drop("Survived",axis=1)

y = train["Survived"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print("F1 Score:",f1_score(y_test, predictions))

 

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)