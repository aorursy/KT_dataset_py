# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

path=os.listdir("../input")



print(path)

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/train.csv')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

import statsmodels.api as sm

import statsmodels.formula.api as smf
df.tail()
survived_class = df[df['Survived']==1]['Pclass'].value_counts()

dead_class = df[df['Survived']==0]['Pclass'].value_counts()

print(survived_class)

print(dead_class)
df_class = pd.DataFrame([survived_class,dead_class])

df_class.head()
df_class.index = ['Survived','Died']

df_class.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Class")
from IPython.display import display

display(df_class)
Survived = df[df.Survived == 1]['Sex'].value_counts()

Died = df[df.Survived == 0]['Sex'].value_counts()

df_sex = pd.DataFrame([Survived , Died])

df_sex.index = ['Survived','Died']

df_sex.plot(kind='bar',stacked=True, figsize=(5,3), title="Survived/Died by Sex")
display(df_sex)
survived_embark = df[df['Survived']==1]['Embarked'].value_counts()

dead_embark = df[df['Survived']==0]['Embarked'].value_counts()

df_embark = pd.DataFrame([survived_embark,dead_embark])

df_embark.index = ['Survived','Died']

df_embark.plot(kind='bar',stacked=True, figsize=(5,3))
display(df_embark)
X = df.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)

y = X.Survived                       # vector of labels (dependent variable)

X=X.drop(['Survived'], axis=1)       # remove the dependent variable from the dataframe X



X.head()


# import seaborn as sns

# relation = X.corr()

# print(relation)

# sns.pairplot(X)
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

X.Sex=labelEncoder_X.fit_transform(X.Sex)
df['Embarked'].unique()
X['Embarked'].mode()
X['Embarked'].fillna('S', inplace = True) 
X['Embarked'].unique()
X['Embarked'].head()
integer_encoded = labelEncoder_X.fit_transform(X['Embarked'])

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
onehot_encoded
X['Embarked_C']=onehot_encoded[:,1]

X['Embarked_Q']=onehot_encoded[:,2]

X=X.drop(['Embarked'], axis=1) 
X['Age'].isnull().sum()



X['Age'].fillna(X['Age'].median(), inplace = True) 
X['Age'].isnull().sum()
X=X.drop(['Name'], axis=1)

X.head()
# n_traning= df.shape[0] 

# n_traning
# #converting all value above eighteen as zero(old people) and below eighteen as one(young)

# for i in range(0, n_traning):

#     if X.Age[i] > 18:

#         X.Age[i]= 0

#     else:

#         X.Age[i]= 1
X.head()

#to see the correlation relation between features

import seaborn as sns

relation = X.corr()

print(relation)

sns.pairplot(X)
#-----------------------Logistic Regression---------------------------------------------

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(penalty='l2',random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)
print("Logistic Regression:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("Random Forest:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std())
# Fitting Kernel SVM to the Training set

from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X=X , y=y , cv = 10)

print("SVM:\n Accuracy:", accuracies.mean(), "+/-", accuracies.std(),"\n")
df_test=pd.read_csv('../input/test.csv')
df_test.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 6)

X_train.head()
'''we can see the cross validation is giving random forrest with best result so 

i'm using that model for prediction'''



from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train ,y_train)
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
X_test.head()
X_test['y_pred']=y_pred
X_test.head()
X['survived']=y
X.head()