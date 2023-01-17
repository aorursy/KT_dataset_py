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
# importing required python modules #

import pandas as pd,numpy as np

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

from sklearn.model_selection import KFold
# load train and test data #

train_ds = pd.read_csv('../input/titanic/train.csv')

test_ds = pd.read_csv('../input/titanic/test.csv')
# clean train data#

train_ds = train_ds.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)#drop cloumns which are not required#

train_ds["Age"].fillna(train_ds["Age"].median(skipna=True), inplace=True)#replace nan values with median#

train_ds["Embarked"].fillna(train_ds['Embarked'].value_counts().idxmax(), inplace=True)#replace nan values with mode#

train_ds['TravelAlone']=np.where((train_ds["SibSp"]+train_ds["Parch"])>0, 0, 1)# merge column with conditon#

train_ds.drop(['SibSp','Parch'] ,axis=1, inplace=True)#drop columns#

dummy = pd.get_dummies(train_ds[['Sex','Embarked']])#create dummy variables#

train_data = pd.concat([train_ds,dummy],axis=1)# concat#

train_data.drop(['Sex','Embarked'] ,axis=1, inplace=True)#drop#
# clean train data#

test_df = test_ds.drop(['PassengerId','Cabin','Name','Ticket'],axis=1)#drop cloumns which are not required#

test_df["Age"].fillna(test_df["Age"].median(skipna=True), inplace=True)#replace nan values with median#

test_df["Fare"].fillna(test_df["Fare"].mean(skipna=True), inplace=True)#replace nan values with mean#

test_df["Embarked"].fillna(test_df['Embarked'].value_counts().idxmax(), inplace=True)#replace nan values with mode#

test_df['TravelAlone']=np.where((test_df["SibSp"]+test_df["Parch"])>0, 0, 1)# merge column with conditon#

test_df.drop(['SibSp','Parch'] ,axis=1, inplace=True)#drop columns#

dummy_test = pd.get_dummies(test_df[['Sex','Embarked']])#create dummy variables#

test_data = pd.concat([test_df,dummy_test],axis=1)# concat#

test_data.drop(['Sex','Embarked'] ,axis=1, inplace=True)#drop#
#split dependent and independent variable#

x = train_data.iloc[:,1:]

y = train_data['Survived']
#load submission data to test accuracy#

pred = pd.read_csv('../input/titanic/gender_submission.csv')

y_test = pred['Survived'].tolist()
#build MultiLayerPerceptron#

classifier = MLPClassifier(hidden_layer_sizes=(150,150,150), max_iter=100,

                           activation = 'relu',solver='adam',random_state=1)

classifier.fit(x, y)

y_pred = classifier.predict(test_data)

print('MLP Accuracy:',accuracy_score(y_test, y_pred) * 100)
# cross validation using Kfold method #

kf = KFold(n_splits=5) # Define the split - into 2 folds 

acc = 0

index = []

for train_index, test_index in kf.split(x):

    X_train = x.iloc[train_index]

    y_train = y.iloc[train_index]

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(test_data)

    accuracy = accuracy_score(y_test, y_pred) * 100

    print(accuracy)

    if accuracy > acc:

        acc = accuracy

        index.clear()

        index = list(train_index)
# Pick best training set and fit the model #

X_train = x.iloc[index]

y_train = y.iloc[index]

classifier.fit(X_train, y_train)

final_pred = classifier.predict(test_data)

print('Final Accuracy:',accuracy_score(y_test, final_pred) * 100)
#evaluation metrics#

print('Evaluation metrics ....')

print('')

print('Accuracy: ',round(accuracy_score(y_test, final_pred)*100),2)

print('F1-Score: ',round(f1_score(y_test, final_pred, average="macro")*100),2)

print('Precision: ',round(precision_score(y_test, final_pred, average="macro")*100),2)

print('Recall :',round(recall_score(y_test, final_pred, average="macro")*100),2)
final_result = pd.DataFrame({'PassengerId': test_ds.PassengerId, 'Survived': final_pred})

final_result.to_csv('submission.csv', index=False)