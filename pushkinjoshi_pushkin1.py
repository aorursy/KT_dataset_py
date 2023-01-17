import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

%matplotlib inline
url="../input/Churn_6SIJGngxq2.csv"
df=pd.read_csv("../input/Churn_6SIJGngxq2.csv") 

Total_Users = len(df)
print('Total ', Total_Users)
print (df.loc[:,'churned'])
df.describe()
df.fillna(value = df.median(), inplace = True)
df.describe()
Total = len(df)
print('Total ', Total)
if (Total==5000):
    print ('yes')
else:
    print ('no')
X = df.copy() # copies the dataframe to X
y = df.churned # copis the churned column to y
del X['churned'] # deletes the churned column from X
print (df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, train_size=0.7)
print(df)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
