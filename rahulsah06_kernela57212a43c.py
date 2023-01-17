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
#importing pakages

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
#importing dataset

dataset = pd.read_csv('/kaggle/input/machine-learning-for-diabetes-with-python/diabetes_data.csv')
dataset.head()
dataset.shape
dataset.info()
sns.heatmap(dataset.isnull())
dataset.groupby('Outcome').hist(figsize=(9,9))
#splitting the independent and dependent variables.

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,8].values
#counting the outcome of the diabetes study.

sns.countplot(dataset['Outcome'],label='Count')
print(dataset.groupby('Outcome').size())
#importing train test split

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(dataset.iloc[:,dataset.columns != 'Outcome'],

                                                dataset['Outcome'],stratify=dataset['Outcome'],random_state=66)
#importing KNN classifier

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []

test_accuracy = []

neighbors_settings = range(1,100)

for n_neighbors in neighbors_settings:

    knn = KNeighborsClassifier(n_neighbors= n_neighbors)

    knn.fit(X_train, Y_train)

    training_accuracy.append(knn.score(X_train,Y_train))

    test_accuracy.append(knn.score(X_test,Y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')

plt.plot(neighbors_settings, test_accuracy, label='test accuracy')

plt.xlabel('Accuracy')

plt.ylabel('n_neighbors')

plt.legend()

plt.show()
#k value can be taken as 19 as seen from the above graph

knn = KNeighborsClassifier(n_neighbors=19)

knn.fit(X_train, Y_train)

print('Accuracy of KNN classifier on training set: {:.2f}'.format(knn.score(X_train, Y_train)))

print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, Y_test)))
#importing the confusion matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

actual =Y_test

predicted =knn.predict(X_test)

results = confusion_matrix(actual, predicted)

print('Confusion Matrix')

print(results)

print('Accuracy Score :', accuracy_score(actual, predicted))

print('Report')

print(classification_report(actual,predicted))
#importing roc curve and roc curve score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
def plot_roc_curve(fpr,tpr):

  plt.plot(fpr, tpr, color = 'orange', label = 'ROC')

  plt.plot([0,1], [0,1],color='darkblue',linestyle='--')

  plt.title('Receiver Operating Charactersticks (ROC) Curve')

  plt.xlabel('False Positive Value')

  plt.ylabel('True Positive Rate')

  plt.legend()

  plt.show()
#predicting the probablity by KNN classifier

probs = knn.predict_proba(X_test)

probs[0:10]
probs = probs[:,1]

probs[0:10]
auc = roc_auc_score(Y_test,probs)

print('AUC: %.2f' %auc)
fpr, tpr, thresholds = roc_curve(Y_test, probs)
#plotting the roc curve

plot_roc_curve(fpr, tpr)