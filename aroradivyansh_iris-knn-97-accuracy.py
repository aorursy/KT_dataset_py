import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

model=KNeighborsClassifier()
import pandas as pd

data = pd.read_csv("../input/iris/Iris.csv")
data.head()
#changing column names accordingly

data.columns=["id" , 

             "SepalLengthCm",

             "SepalWidthCm" ,

             "PetalLengthCm" ,

              "PetalWidthCm" ,

              "Species"]

data.head()                    
# X= given 

# y= to be founded

X = data.drop(["Species"] , axis = 1)   #axis=1 removes Species column

y = data["Species"]

print(X.shape,y.shape)
print(X,y)


#Data split into train and test



X_train ,X_test , y_train , y_test =train_test_split(X,y,test_size=.25,random_state=100)

print ("Shapes of X_train, X_test , y_train , y_test are :" )

print("       " , X_train.shape , X_test.shape , y_train.shape , y_test.shape)
X_test.head()
print(model)
model.fit(X_train , y_train)
till_row = 20

temp = X_test[:till_row]

temp["Species"] = y_test[ :till_row]

temp["predicted"] = model.predict(X_test[ : till_row])

temp
acc = model.score(X_test,y_test)

print(acc)

print(acc*100)
predicted = model.predict(X_test)

original = y_test.values

wrong = 0



for i in range (len(predicted)):

  if predicted[i]!=original[i]:

    wrong = wrong + 1

print(wrong)