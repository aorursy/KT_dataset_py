import pandas as pd

data=pd.read_csv('../input/flight-route-database/routes.csv')

nan=data[data.isnull().any(axis=1)].head()

nan.iloc[0:3]
data2=data.fillna(0)

data2.head()
import matplotlib.pyplot as plt

team=['CSK','KKR','DC','MI']

score=[163,180,156,162]

plt.bar(team,score,color=['red','black','red','red'])

plt.title('IPL TEAM SCORE GRAPH')

plt.xlabel('TEAMS')

plt.ylabel('SCORE')
import numpy as np

t1=np.array([1,2,3,4,5,6,7,8,9,10])

t2=np.array([5,6,10])

res=[m for m, val in enumerate(t1) if val in set(t2)]

new_arr=np.delete(t1,res)

print("arr1:",t1)

print("arr2:",t2)

print("newarr:",new_arr)

print("arr2:",t2)
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