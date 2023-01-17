import pandas as pd

data=pd.read_csv('../input/flight-route-database/routes.csv')

nan=data[data.isnull().any(axis=1)].head()

nan.iloc[0:3]
data2=data.fillna(0)

data2.head()
import matplotlib.pyplot as plt

team=['CSK','KKR','DC','MI']

score=[154,198,161,124]

      

plt.bar(team,score,color=['GREEN','red','GREEN','GREEN'])

plt.title('IPL TEAM SCORE GRAPH')

plt.xlabel('TEAMS')

plt.ylabel('SCORE')
import numpy as np

import pandas as pd



a = np.array([1,2,3,4,5,6])



b = np.array([2,4,6])



c = np.intersect1d(a,b) #printing the common items



print(c)

print("\n")

for i in b:

    for j in a:

        if i == j:

            a = a[a!=j] #removing the common items from the array "a"

print(a)

print("\n")
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