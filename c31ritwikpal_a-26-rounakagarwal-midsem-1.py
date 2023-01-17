import pandas as pd

import numpy as np

data=pd.read_csv("../input/flight-route-database/routes.csv")

print(data.head(11))

print(data[data.isnull().any(axis=1)].head(1))

data.replace(to_replace=np.nan,value=0)
import matplotlib.pyplot as plt

team=['CSK','KKR','DC','MI']

score=[146,184,157,175]

    

      

plt.bar(team,score,color=['green','red','green','green'])

plt.title('IPL')

plt.xlabel('TEAMS')

plt.ylabel('SCORE')
a = np.array([1,8,2,6,4,9])

b = np.array([1,3,6,5])



c = np.intersect1d(a,b) #Finding the common items



print("Common items are: ",c)

print("\n")

for i in b:

    for j in a:

        if i == j:

            a = a[a!=j] #removing the common items from the array "a"

print(" 1st array:",a)

print("\n")

print(" 2nd array:",b)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



train = pd.read_csv("../input/iris/Iris.csv")





X = train.drop("Species",axis=1)

y = train["Species"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



print("F1 Score:",f1_score(y_test, predictions,average='weighted'))

 

print("\nConfusion Matrix(below):\n")

confusion_matrix(y_test, predictions)