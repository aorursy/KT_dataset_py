import pandas as pd

from sklearn import tree

import numpy as np

%matplotlib inline
data = pd.read_csv('../input/Iris.csv',header=0)
data.head()
data.info()
data['SepalLengthCm'] = data['SepalLengthCm'].astype(int)
data['SepalLengthCm'] = data['SepalLengthCm'].astype(int)

data['SepalWidthCm'] = data['SepalWidthCm'].astype(int)

data['PetalLengthCm'] = data['PetalLengthCm'].astype(int)

data['PetalWidthCm'] = data['PetalWidthCm'].astype(int)
##mapping the species

d = {'Iris-setosa' : 1, 'Iris-versicolor' : 2, 'Iris-virginica' : 3}

data['Species'] = data['Species'].map(d)
data['Species']=data['Species'].astype(int)
features = list(data.columns[1:5])

features
y = data["Species"]

x = data[features]

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(x,y)

output = Tree.predict([5.4,2,4.5,1])

print (output)
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 



# Create the random forest object which will include all the parameters

# for the fit

forest = RandomForestClassifier(n_estimators = 100)



# Fit the training data to the Survived labels and create the decision trees

forest = forest.fit(x,y)



# Take the same decision trees and run it on the test data

output = forest.predict([5.4,2,4.5,1])



print (output)