import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

User = pd.read_csv("../input/user-data/User_Data.csv")
User.head()
x = User.iloc[:, [2,3]].values 

y = User.iloc[:, 4].values 
from sklearn.model_selection import train_test_split 

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25) 



from sklearn.preprocessing import StandardScaler 

Haha = StandardScaler() 

xtrain = Haha.fit_transform(xtrain)  

xtest = Haha.transform(xtest) 

  

print (xtrain[0:10, :]) 



print (xtest[0:10, :]) 


from sklearn.linear_model import LogisticRegression 

cl = LogisticRegression() 

cl.fit(xtrain, ytrain) 

print("The coefficient of the model are",cl.coef_)





inter=cl.intercept_

print("The intercept of the model is : \n",inter)
y_pred = cl.predict(xtest)
print("The predictions of the Logistic trained model are:\n",y_pred)


from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(ytest, y_pred) 

  

print ("Confusion Matrix : \n", cm) 



from sklearn.metrics import accuracy_score 

print ("The Accuracy of the Model  : ", accuracy_score(ytest, y_pred)) 
