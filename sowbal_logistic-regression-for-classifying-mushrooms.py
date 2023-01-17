# Load some standard libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Read input

inputdata = pd.read_csv("../input/mushrooms.csv")

inputdata.head(3)

#for logisitic regression we need numbers instead of letters

from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()



for col in inputdata.columns:

    inputdata[col] = encoder.fit_transform(inputdata[col])

    

inputdata.head(3)
# split into train and test data

from sklearn.model_selection import train_test_split



(train,test) = train_test_split(inputdata, test_size=0.3)

train_output = train['class']

train_input = train[[x for x in train if 'class' not in x]]



test_output = test['class']

test_input = test[[x for x in test if 'class' not in x]]



train_input.head(3)
# apply logistic regression

from sklearn.linear_model import LogisticRegression



logistic_regression = LogisticRegression()

logistic_regression.fit(train_input, train_output)



predicted_output = logistic_regression.predict(test_input)

# check validity



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_output, predicted_output)



print ("Accuracy of Logistic Regression is %f" %accuracy)