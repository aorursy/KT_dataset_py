import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
titanictrain=pd.read_csv("../input/titanic/train.csv")
titanictest=pd.read_csv("../input/titanic/test.csv")
titanictrain.head()
titanictest.head()
titanictrain.hist(column="Age")
titanictest.hist(column="Age")
titanictest.describe()
titanictrain.describe()
titanictrain.info()
titanictest.info()
from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()
x_train=titanictrain[['PassengerId','Pclass','Age']]
y_train=titanictrain['Survived']
x_test=titanictest[['PassengerId','Pclass','Age']]
lm.fit(x_train,y_train)
y_train
predictions=lm.predict(x_test)

predictions
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,predictions)
print(confusion_matrix)