import pandas as pd #To load pandas 

import numpy as np #To load numpy
#for loading csv file into pandas dataframe

data = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

data.head()
#To convert categorical features into numerical

species={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}

data=data.replace({'species':species})

data.head()
#To get dimensions(shape) of a dataframe

data.shape
#To drop the target column from x dataframe and store it in a different dataframe

x_train=data.drop('species',axis=1)

y_train=data[['species']]
#splits the data into 80% and 20% for training and cross-validation purpose

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x_train,y_train,test_size=0.2,random_state=18)
#Create object/instance of logistic regression classifier available at sklearn(Just like STL in CPP)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()
#Fit the above object with the training data

clf.fit(x_train,y_train)
#Check accuracy on the cross validation data we created using split

clf.score(x_test,y_test)
#To predict values on a dataframe

prediction_test = clf.predict(x_test)
#Conversion to int datatype

prediction_test=prediction_test.astype(int)

print(prediction_test.shape)
#Ignore the statement below. This dataset does not have Id column but ideally you will have the column present

Id = np.random.rand(30,1)

#You will do something like this: 

#Id = data["id"]
#Create a dataframe with ID and Target

my_solution = pd.DataFrame(prediction_test, Id, columns = ["Target"])

print(my_solution.shape)
my_solution.to_csv("prediction.csv", index_label = ["id"])