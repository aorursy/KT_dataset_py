import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
train = pd.read_csv(r"../input/titanic/train.csv")
train
train.head()
sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
sns.set_style('whitegrid')

sns.countplot(x ='Survived', data = train,palette='BuGn_r')
sns.set_style('whitegrid')

sns.countplot(x = train['Survived'],hue = train['Sex'], palette='BuGn_r')
sns.set_style('whitegrid')

sns.countplot(x = train['Pclass'],hue = train['Sex'], palette='BuGn_r')
#plt.figure(figsize = 10,5)

sns.violinplot( x= 'Pclass', y = 'Age',data = train)
median= train['Age'].median()

train['Age'].fillna(median, inplace=True)
median
train['Age'].isnull()
sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
train.drop('Cabin', axis = 1, inplace = True)
train
sns.heatmap(train.isnull(),yticklabels = False, cbar = False, cmap = 'viridis')
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 

                                                 train['Survived'], test_size=0.25, 

                                                   random_state=101)
from sklearn.naive_bayes import GaussianNB
naivemodel = GaussianNB()
naivemodel = naivemodel.fit(X_train, y_train)
naivemodel
Prediction = naivemodel.predict(X_test)
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
matrix = (confusion_matrix(y_test,Prediction))
matrix
print(classification_report(y_test,Prediction))
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test,Prediction)*100)

print("Precision:",metrics.precision_score(y_test,Prediction)*100)

print("Recall:",metrics.recall_score(y_test,Prediction)*100)

print("F1 Score:",metrics.f1_score(y_test,Prediction)*100)
from sklearn.metrics import roc_curve, roc_auc_score
nv_auc= print('roc_auc_score for naive bayes: ', roc_auc_score(y_test, Prediction))
Prediction
submission = pd.DataFrame({'PassengerId':X_test['PassengerId'],'Survived':Prediction})
submission.head()
filename = 'Titanic Predictions by naive-bayes model.csv'
print('Saved file: ' + filename)