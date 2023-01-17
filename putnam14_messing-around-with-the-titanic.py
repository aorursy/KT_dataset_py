# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
trainData = pd.read_csv('/kaggle/input/titanic/train.csv')
testData = pd.read_csv('/kaggle/input/titanic/test.csv')

print(len(trainData) + len(testData))
trainData.fillna(-99999,inplace=True)
testData.fillna(-99999,inplace=True)
testData.head()

def getCabinLetter(cabin):
    cabinStr = str(cabin)
    if cabinStr[0].isalpha():
        return cabinStr[0]
    return -9999

testData['CabinLetter'] = getCabinLetter(testData['Cabin'])
trainData['CabinLetter'] = getCabinLetter(trainData['Cabin'])

testData['Age'] = (testData['Age']).astype(int)
trainData['Age'] = (trainData['Age']).astype(int)

testData['Fare'] = (testData['Fare'] * 100).astype(int)
trainData['Fare'] = (trainData['Fare'] * 100).astype(int)

testData.head()
from sklearn.model_selection import train_test_split

trainDataTrain, trainDataTest = train_test_split(trainData, test_size=0.2, random_state=42, shuffle=True)
'''from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 1)

features = ['Pclass','Sex','SibSp','Parch', 'Age']'''
from catboost import CatBoostClassifier, Pool

model = CatBoostClassifier(iterations=2,
                           depth=2,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)

features = ['Pclass','Sex','SibSp','Parch','Age','CabinLetter','Cabin','Fare','Embarked']
y = trainDataTrain.Survived
x = trainDataTrain[features]
xTest = trainDataTest[features]

model.fit(x, y, features)

predictions = model.predict(xTest)

trainingOutput = pd.DataFrame({'PassengerId': trainDataTest.PassengerId, 'Survived': predictions})
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(trainDataTest['Survived'],trainingOutput['Survived'])

tn, fp, fn, tp = conf_matrix.ravel()

total_answers = tn + fp + fn + tp

accuracy =  (tp + tn) / total_answers
precision = tp / (tp + fp) 
recall = tp / (tp + fn)

true_positive_rate = recall
false_positive_rate = fp / (fp + tn)

f1 = 2/(1/precision + 1/recall )
conf_matrix

# confusion matrix= [[TrueNegative,FalsePositive],
#                     [FalseNegative,TruePositive]]
print('acc:',round(accuracy,4),
      '\nprecision:',round(precision,4),
      '\nrecall or true_positive_rate:',round(recall,4),
      '\nfalse_positive_rate:',round(false_positive_rate,4),
      '\nf1:',round(f1,4))

## high precision: less false positives - most of predicted "ones" are real "ones"
## low recall: more false negatives - more predict "zeros" that actually are "ones"
y = trainData.Survived
x = trainData[features]

model.fit(x, y, features)
xTest = testData[features]

predictions = model.predict(xTest)

finalOutput = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': predictions})
finalOutput.to_csv('my_submission.csv', index=False)

print('CSV ready for submission!')