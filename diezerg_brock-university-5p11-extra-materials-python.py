# load Iris data from web

import pandas as pd

data = pd.read_csv('../input/Iris.csv')  



# read data from csv file, you can change '../input/Iris.csv' into your path for example ('c:/data/abc.csv')

print(data.columns)  # show the titles of the table
data.head()
data.drop('Id',axis=1, inplace=True) # delete useless column 'Id'
data.plot() # simple visualization
from pandas.plotting import scatter_matrix



scatter_matrix(data) # show the relationship between columns
data1 = pd.get_dummies(data) # create dummpy variable

data1 = data1.sample(frac=1) # random shuffle the data

data1.head(10) # show first 10 columns
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix



x_train,  x_validate, y_train, y_validate = train_test_split(data.iloc[:,:4], data.iloc[:,4:], test_size=0.3) 



# split the data into train set and validate set, x is input varriables, y is the target value
y_train.columns # Check the target value name
model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10, min_samples_split=5) # create a decision tree model

model.fit(x_train, y_train) # train the model on train set data

y_return = model.predict(x_train)  # get model result from x_train

print('accuracy: ', accuracy_score(y_train, y_return))  # show the train score

y_predict = model.predict(x_validate) # predict on validate data

print('validate accuracy: ', accuracy_score(y_validate, y_predict))  # show the score on validate data set
print(confusion_matrix(y_validate, y_predict))  # Show the confusion_matrix