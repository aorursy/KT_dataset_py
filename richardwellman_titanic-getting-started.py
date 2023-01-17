import numpy as np 
import pandas as pd 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import os
print(os.listdir("../input"))
df_raw = pd.read_csv('../input/train.csv', low_memory=False)
df_raw.head(5)
df_drop = df_raw.drop(columns=['Name','Ticket','Cabin','PassengerId'])
df_drop.head(5)

df_dummies = pd.get_dummies(df_drop,columns=['Pclass','Sex','Embarked'])
df_dummies.head(5)
df_target = df_dummies['Survived']
df_data = df_dummies.drop(columns=['Survived'])
# fill in NaN values
df_data.fillna(df_data.mean(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, random_state=0)
df_data.head(5)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
print("Test set score: {:.2f}".format(forest.score(X_test, y_test)))
df_test = pd.read_csv('../input/test.csv', low_memory=False)
df_test_drop = df_test.drop(columns=['Name','Ticket','Cabin','PassengerId'])
df_test_dummies = pd.get_dummies(df_test_drop,columns=['Pclass','Sex','Embarked'])
# fill in NaN values
df_test_dummies.fillna(df_test_dummies.mean(), inplace=True)
predictions = forest.predict(df_test_dummies)

predictions
submission = pd.DataFrame({'Passengerid': df_test.PassengerId, 'Survived': predictions})
#submission.to_csv("../output/titanicsubmit.csv", index=False)
submission.head(10)
