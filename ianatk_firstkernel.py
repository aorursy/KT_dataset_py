#Load Modules

import numpy as np

import pandas as pd
#Load Data

data = pd.read_csv('../input/train.csv')

data.head()
#Missing values

data.isna().any()
#Replace Missing values and remove some columns

data.drop(columns = ['PassengerId','Name','Ticket','Cabin'],inplace=True)

data.Age.fillna(data.Age.mean(),inplace=True)

data.head()
#Dummy Variables, feature and target set

X = pd.get_dummies(data.drop(columns=['Survived'])).values

y = data.Survived.values
#Split into Train and Test Set, and Random Forest Classification

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)



from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100)

model.fit(X_train,y_train)



#Prediction

y_pred = model.predict(X_test)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)

print('Accuracy: ',round(cm.trace()/cm.sum(),2))

print('False negatives: ',round(cm[1,0]/cm[1,:].sum(),2))