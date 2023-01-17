import pandas as pd

import numpy as np
#displaying the first 11 rows

df=pd.read_csv("../input/titanic/train_and_test2.csv")

df.head(11)
#displaying a row having a missing value

null_data = df[df.isnull().any(axis=1)]

null_data.head(1)
#replacing missing values by NaN

df=df.fillna("NaN")

print(df.isnull().values.ravel().sum()) #No more missing values
import matplotlib.pyplot as plt



team=['CSK','KKR','DC','MI']

score=[154,198,161,124]

      

plt.bar(team,score,color=['GREEN','RED','GREEN','GREEN'])

plt.title('IPL TEAM SCORE GRAPH')

plt.xlabel('TEAMS')

plt.ylabel('SCORE')

plt.show()
a = np.array([1,2,3,4,5,6])

b = np.array([2,4,6,8,10])



c = np.intersect1d(a,b) #Finding the common items



print("Common items are: ",c)

print("\n")

for i in b:

    for j in a:

        if i == j:

            a = a[a!=j] #removing the common items from the array "a"

print("Removing items from 1st array:",a)

print("\n")

print("Items in 2nd array:",b)
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