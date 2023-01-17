# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/byudatasciencecapstone/magic04.csv')

data.info()
from sklearn.model_selection import train_test_split



X = data.iloc[:,:len(data.columns)-1]

Y = data['class']



X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0, test_size = 0.3)

X_train = X_train.to_numpy()

X_test = X_test.to_numpy()
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train, y_train)



print("Training Accuracy:", lr.score(X_train, y_train)) # Accuracy of the model when training.

print("Testing Accuracy:", lr.score(X_test, y_test) ) # Accuracy of the test.
y_pred = lr.predict_proba(X)

mySubmission = pd.DataFrame({'Id': range(len(data)), 'Predicted': y_pred[:,1]})

mySubmission.head()



# Use the following line to save your submisstion. You can use any filename. 

# mySubmission.to_csv('Magic04Submission.csv', index=False)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score



xgb = XGBClassifier()

xgb.fit(X_train, y_train)



y_pred = xgb.predict(X_train)

train_predictions = [round(value) for value in y_pred]



y_pred = xgb.predict(X_test)

test_predictions = [round(value) for value in y_pred]

    

print("Training Accuracy:",accuracy_score(train_predictions, y_train)) #Accuracy of the model when training.

print("Testing Accuracy:", accuracy_score(test_predictions, y_test)) # Accuracy of the test.
y_pred = xgb.predict_proba(X.to_numpy())



mySubmission = pd.DataFrame({'Id': range(len(data)), 'Predicted': y_pred[:,1]})

mySubmission.head()



# Use the following line to save your submisstion. You can use any filename. 

# mySubmission.to_csv('Magic04XGBClassifierSubmission.csv', index=False)
%matplotlib inline

import matplotlib.pyplot as plt

from sklearn import metrics



fpr, tpr, thresholds = metrics.roc_curve(Y.to_numpy(), y_pred[:,1])

auc = metrics.auc(fpr, tpr)





plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) curve')

plt.legend(loc="lower right")

plt.show()