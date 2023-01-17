1.
import pandas as pd
data=pd.read_csv('../input/flight-route-database/routes.csv')
nan=data[data.isnull().any(axis=1)].head()
nan.iloc[0:3]

data2=data.fillna(0)
data2.head()
2.
import matplotlib.pyplot as plt
team=['CSK','KKR','DC','MI']
score=[157,179,165,136]
      
plt.bar(team,score,color=['yellow','red','yellow','yellow'])
plt.title('IPL TEAM SCORE GRAPH')
plt.xlabel('TEAMS')
plt.ylabel('SCORE')
3.
import numpy as np
arr1=np.array([10,20,30,40,50,60,70,80,90,100])
arr2=np.array([50,70,90])
r=[m for m, val in enumerate(arr1) if val in set(arr2)]
new_arr=np.delete(arr1,r)
print("array1:",arr1)
print("array2:",arr2)
print("newarray:",new_arr)
print("array2:",arr2)
4.
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