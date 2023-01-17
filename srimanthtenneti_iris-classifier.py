import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
datasets = pd.read_csv("../input/iris/Iris.csv")
datasets
target = datasets["Species"] 

target.astype('category') 
target_encode = [] #List to store all the encoded data

for i in range(len(target)):

  if (target[i] == "Iris-setosa"): # 0

    target_encode.append(0)

  elif (target[i] == "Iris-versicolor"): # 1

    target_encode.append(1)

  else:

    target_encode.append(2) # 2
target_encode[:10]
data = datasets["SepalLengthCm"] # X
plt.plot(data)

plt.title("Data")
x = data

y = target_encode



x = np.array(x)

y = np.array(y)
from sklearn.model_selection import train_test_split #Split the dataset into parts

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) #30% data -> Test data 
poly = PolynomialFeatures(14).fit_transform(X_train.reshape(-1 , 1)) #Quadratic Regression -> X_train || y_train

reg2 = LinearRegression()

reg2.fit(poly , y_train) 
plt.plot(y_train) # Ground Truth or Actual Value

plt.plot(reg2.predict(poly)) # Y_Pred
pred_target = np.array([7.9]) # Iris-virginica sepal length
poly1 = PolynomialFeatures(14).fit_transform(pred_target.reshape(1 , -1)) #Quadratic Regression -> X_train || y_train
pred = np.round (reg2.predict(poly1))
def predict(pred):

  if (pred == 0):

    return ("Iris-setosa")

  elif (pred == 1):

    return ("Iris-versicolor")

  else:

    return ("Iris-virginica")
predict(pred)