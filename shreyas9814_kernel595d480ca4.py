import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import matplotlib as pt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_absolute_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error





from sklearn.model_selection import train_test_split



train = pd.read_csv("train.txt",sep=" |,",header=None)



test = pd.read_csv("test.txt",sep=" |,",header=None)



label=train["class"]

features=train.drop(["class","time"],axis=1)



X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2,random_state=0)



from sklearn.preprocessing import StandardScaler  

scaler = StandardScaler()  

scaler.fit(X_train)  

X_train = scaler.transform(X_train)  

X_test = scaler.transform(X_test)



clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)





test_X = test[features.columns]



test_predictions = clf.predict(X_test)

acc = accuracy_score(y_test, test_predictions)

print("Accuracy: ",acc)





"""output = pd.DataFrame({'time': test.time,

                      'class': test_predictions})

output.to_csv('submission.csv', index=False)"""