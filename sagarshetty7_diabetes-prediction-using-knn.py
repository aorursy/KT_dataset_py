#importing pakages

%matplotlib inline

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

diabetes = pd.read_csv("../input/diabetes.csv")

diabetes.head()
diabetes.shape
diabetes.info()
sns.heatmap(diabetes.isnull())
diabetes.groupby('Outcome').hist(figsize=(10,9))

# from below graphs we can visualize details like

#for non diabetes age is lies mostly 20 to 40

#for diabetes age is lies mostly 20 to 60 for

#for non diabetes and diabetes there is no change in BMI,Blood Pressure,Diabetes PedigreeFunction 

#for diabetes Glucose is more towards 200 compared to non diabetes etc
print(diabetes.groupby('Outcome').size()) #1 means diabetes. Of these 768 data points, 500 are labeled as 0 and 268 as 1:

#counting the outcome of the diabetes study.

sns.countplot(diabetes['Outcome'],label='Count')
from sklearn.model_selection import train_test_split

X_train ,X_test ,y_train , y_test = train_test_split( diabetes.loc[:,diabetes.columns !='Outcome'], diabetes['Outcome'],

                                                     stratify=diabetes['Outcome'],random_state=66)
#importing KNN classifier

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []

test_accuracy = []

neighbors_settings = range(1,100)

for n_neighbors in neighbors_settings:

    knn = KNeighborsClassifier(n_neighbors= n_neighbors)

    knn.fit(X_train, y_train)

    training_accuracy.append(knn.score(X_train,y_train))

    test_accuracy.append(knn.score(X_test,y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')

plt.plot(neighbors_settings, test_accuracy, label='test accuracy')

plt.xlabel('Accuracy')

plt.ylabel('n_neighbors')

plt.legend()

plt.show()
#k value can be taken as 19 as seen from the above graph

knn = KNeighborsClassifier(n_neighbors=19)

knn.fit(X_train, y_train)

print('Accuracy of KNN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

print('Accuracy of KNN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
#importing the confusion matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

actual =y_test

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
auc = roc_auc_score(y_test,probs)

print('AUC: %.2f' %auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)

#plotting the roc curve

plot_roc_curve(fpr, tpr)