# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
training_data = pd.read_csv("../input/train.csv")
training_data.describe()

training_data.head()
training_data.isnull().sum()
training_data.info()
n = 0
for col in training_data:
    print (training_data.dtypes.index[n])
    if (training_data[col].dtypes) == object:
        if (len(training_data[col].unique())) <= 20:
            print ("Encodable - len =", len(training_data[col].unique()))
    else:
        print ("Should not be encoded")
    n += 1
# Another way to do this is to only print values that should be encoded:
n = 0
for col in training_data:
    if (training_data[col].dtypes) == object:
        if (len(training_data[col].unique())) <= 20:
            print (training_data.dtypes.index[n])
            print ("Encodable - len =", len(training_data[col].unique()))
    n += 1
# For reference, here is the len of all dtype==object columns.
n = 0
for col in training_data:
    if (training_data[col].dtypes) == object:
        print (training_data.dtypes.index[n])
        print ("Encodable - len =", len(training_data[col].unique()))
    n += 1
training_data2 = training_data.copy()
training_data2['female'] = training_data2.Sex == 'female'
training_data2.head()
training_data3 = training_data2.copy()
training_data3['child'] = training_data2.Age < 16
training_data3.head()
training_data4 = training_data3.copy()
training_data4['Age'] = training_data3.Age.fillna(training_data3['Age'].median())
training_data4.isnull().sum()
# here are the models I'll use for a first-try
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# here are the metrics I'll check them with
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
# and the code to split the test/train data
from sklearn.model_selection import train_test_split
# I'm going to write a function to make the confusion matrix easier on the eyes
def confusionMatrixBeautification(y_true, y_pred):
    rows = ['Actually Died', 'Actually Lived']
    cols = ['Predicted Dead', 'Predicted Lived']
    conf_mat = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(conf_mat, rows, cols)
# I also want to split the training_data dataframe into a training and testing portion
train_baseline, test_baseline = train_test_split(training_data4, random_state = 0)
train_baseline.shape, test_baseline.shape
# For a first run, I'll try using the following features
features = ['Age', 'female', 'child', 'Fare', 'Pclass']
target = 'Survived'
# Now comes fitting the models
model01 = RandomForestClassifier(random_state=0)
model02 = DecisionTreeClassifier(random_state=0)
model03 = LogisticRegression()

model01.fit(train_baseline[features], train_baseline[target]);
model02.fit(train_baseline[features], train_baseline[target]);
model03.fit(train_baseline[features], train_baseline[target]);
# Now let's define a pretty function to measure the scores of those models
def printScore(model_number):
    print("Train Accuracy: ", round(accuracy_score(train_baseline[target], model_number.predict(train_baseline[features]))*100,2), "%")
    print("Train Recall: ", round(recall_score(train_baseline[target], model_number.predict(train_baseline[features]))*100,2), "%")
    print("Train Confusion Matrix: \n", confusionMatrixBeautification(train_baseline[target], model_number.predict(train_baseline[features])))
    print("Test Accuracy: ", round(accuracy_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")
    print("Test Recall: ", round(recall_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")
    print("Test Confusion Matrix: \n", confusionMatrixBeautification(test_baseline[target], model_number.predict(test_baseline[features])))
    
def printTestRecall(model_number):
    print("Test Recall: ", round(recall_score(test_baseline[target], model_number.predict(test_baseline[features]))*100,2), "%")
print("RandomForestClassifier()")
printTestRecall(model01)
print("\n\nDecisionTreeClassifier()")
printTestRecall(model02)
print("\n\nLogisticRegression()")
printTestRecall(model03)
# I'll run those same models with all features which are numerical/boolean (except ID)
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'child']
model01.fit(train_baseline[features], train_baseline[target]);
model02.fit(train_baseline[features], train_baseline[target]);
model03.fit(train_baseline[features], train_baseline[target]);
print("RandomForestClassifier()")
printTestRecall(model01)
print("\n\nDecisionTreeClassifier()")
printTestRecall(model02)
print("\n\nLogisticRegression()")
printTestRecall(model03)