import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/mushrooms.csv")

data.head(10)
labelEncoder = preprocessing.LabelEncoder()

for col in data.columns:

    data[col] = labelEncoder.fit_transform(data[col])

    

#print(data)



# Splitting test train set, with 40% of the data as the validation set

train, test = train_test_split(data, test_size = 0.6) 
# Train set

train_y = train['class']

train_x = train[[x for x in train.columns if 'class' not in x]]

# Test/Validation set

test_y = test['class']

test_x = test[[x for x in test.columns if 'class' not in x]]



model = svm.SVC()

model.fit(train_x, train_y)

print ('The accurancy of SVM is ' + str(accuracy_score(test_y, model.predict(test_x))))

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(train_x, train_y)



print ('The accurancy of Decision tree classifier  is ' + str(accuracy_score(test_y, model.predict(test_x))))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis()

model.fit(train_x, train_y)

print ('The accurancy of Linear Discriminant Analysis  is ' + str(accuracy_score(test_y, model.predict(test_x))))