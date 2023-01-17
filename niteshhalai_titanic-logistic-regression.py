import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/titanic/train.csv')
y = train['Survived']

train.drop(labels = ['Survived','PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1, inplace = True)

train['Age'].fillna(train['Age'].mean(), inplace = True)

categorical_columns = ['Sex']

train = pd.get_dummies(train,columns = categorical_columns, dtype = int)

train.drop(labels = ['Sex_male'], axis = 1, inplace = True)



X = []

for column in train.columns:

    X.append(column)



X = train[X]
X.head()
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(X, y)

y_pred = model.predict(train)





unique, counts = np.unique( np.asarray(y_pred == y), return_counts=True)

true_false_values = dict(zip(unique, counts))

accuracy = true_false_values[True]/len(np.asarray(y_pred == y))

accuracy


from sklearn import metrics



cm = metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Confusion Matrix'

plt.title(all_sample_title, size = 15);
original_test = pd.read_csv('/kaggle/input/titanic/test.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

test.drop(labels = ['PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1, inplace = True)

test['Age'].fillna(test['Age'].mean(), inplace = True)

categorical_columns = ['Sex']

test = pd.get_dummies(test,columns = categorical_columns, dtype = int)

test.drop(labels = ['Sex_male'], axis = 1, inplace = True)

test['Fare'].fillna(test['Fare'].mean(), inplace = True)



test_pred = model.predict(test)

#list_of_predictions_test = []



#for pred in test_pred:

#    list_of_predictions_test.append(one_or_zero(pred))

    

#test_pred = np.asarray(list_of_predictions_test)
submission = pd.DataFrame({

        "PassengerId": original_test["PassengerId"],

        "Survived": test_pred

    }) 



filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)