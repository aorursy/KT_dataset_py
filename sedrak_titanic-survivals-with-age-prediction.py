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
import keras
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM, Embedding
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 14, 25, 35, 60, np.inf]
labels = ['Unknown', 'Child', 'Teenager', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()
age_mapping = {'Unknown': None,'Child': 1, 'Teenager': 2, 'Young Adult': 3, 'Adult': 4, 'Senior': 5}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
train = train.fillna({"Embarked": "S"})
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:
   
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms','Countess','Miss','Mme'], 'Mrs')
    dataset['Title'] = dataset['Title'].replace(['Major','Sir','Capt','Col','Don','Jonkheer','Rev','Master','Lady'], 'Mr')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Dr": 1, "Mr": 2, "Mrs": 3}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)
train_age = train
modifiedFlights = train_age.dropna()
null_columns=train.columns[train.isnull().any()]
x_train_age = modifiedFlights.drop(['AgeGroup'], axis = 1)
y_train_age = modifiedFlights["AgeGroup"]
x_test_AgeGroup = train[train.isnull().any(axis=1)]
x_test_age = x_test_AgeGroup.drop(['AgeGroup'], axis = 1)
from sklearn.model_selection import train_test_split

predictors = x_train_age.drop(['PassengerId'], axis=1)
target = y_train_age
x_trainage, x_valage, y_trainage, y_valage = train_test_split(predictors, target, test_size = 0.1, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_trainage, y_trainage)
y_predage = gbk.predict(x_valage)
acc_gbkage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gbkage)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_trainage, y_trainage)
y_predage = sgd.predict(x_valage)
acc_sgdage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_sgdage)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_trainage, y_trainage)
y_predage = knn.predict(x_valage)
acc_knnage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_knnage)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_trainage, y_trainage)
y_predage = randomforest.predict(x_valage)
acc_randomforestage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_randomforestage)
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_trainage, y_trainage)
y_predage = decisiontree.predict(x_valage)
acc_decisiontreeage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_decisiontreeage)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_trainage, y_trainage)
y_predage = perceptron.predict(x_valage)
acc_perceptronage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_perceptronage)
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_trainage, y_trainage)
y_predage = linear_svc.predict(x_valage)
acc_linear_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_linear_svcage)
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_trainage, y_trainage)
y_predage = svc.predict(x_valage)
acc_svcage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_svcage)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_trainage, y_trainage)
y_predage = logreg.predict(x_valage)
acc_logregage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_logregage)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_trainage, y_trainage)
y_predage = gaussian.predict(x_valage)
acc_gaussianage = round(accuracy_score(y_predage, y_valage) * 100, 2)
print(acc_gaussianage)
models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbkage, acc_sgdage, acc_knnage, acc_randomforestage, 
              acc_decisiontreeage, acc_perceptronage, acc_linear_svcage, acc_svcage, acc_logregage, 
               acc_gaussianage]})
models.sort_values(by='Score', ascending=False)
predictions = randomforest.predict(x_test_age.drop('PassengerId', axis=1))
k=0
for i in range(891):
    if np.isnan(train_age['AgeGroup'][i]) == True:
        train_age['AgeGroup'][i] = predictions[k]
        k+=1
test_test_age = test 
modifiedFlights = test_test_age.dropna()
null_columns=test.columns[test.isnull().any()]
x_test_test_age = modifiedFlights.drop(['AgeGroup'], axis = 1)
y_test_test_age = modifiedFlights["AgeGroup"]

x_test_test_AgeGroup = test[test.isnull().any(axis=1)]
x_tst_test_age = x_test_test_AgeGroup.drop(['AgeGroup'], axis = 1)
from sklearn.model_selection import train_test_split

predictors = x_test_test_age.drop(['PassengerId'], axis=1)
target = y_test_test_age
x_testage_1, x_valage_1, y_testage_1, y_valage_1 = train_test_split(predictors, target, test_size = 0.1, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_testage_1, y_testage_1)
y_predage_1 = gbk.predict(x_valage_1)
acc_gbkage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_gbkage_1)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_testage_1, y_testage_1)
y_predage_1 = sgd.predict(x_valage_1)
acc_sgdage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_sgdage_1)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_testage_1, y_testage_1)
y_predage_1 = knn.predict(x_valage_1)
acc_knnage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_knnage_1)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_testage_1, y_testage_1)
y_predage_1 = randomforest.predict(x_valage_1)
acc_randomforestage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_randomforestage_1)
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_testage_1, y_testage_1)
y_predage_1 = decisiontree.predict(x_valage_1)
acc_decisiontreeage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_decisiontreeage_1)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_testage_1, y_testage_1)
y_predage_1 = perceptron.predict(x_valage_1)
acc_perceptronage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_perceptronage_1)
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_testage_1, y_testage_1)
y_predage_1 = linear_svc.predict(x_valage_1)
acc_linear_svcage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_linear_svcage_1)
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_testage_1, y_testage_1)
y_predage_1 = svc.predict(x_valage_1)
acc_svcage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_svcage_1)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_testage_1, y_testage_1)
y_predage_1 = logreg.predict(x_valage_1)
acc_logregage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_logregage_1)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_testage_1, y_testage_1)
y_predage_1 = gaussian.predict(x_valage_1)
acc_gaussianage_1 = round(accuracy_score(y_predage_1, y_valage_1) * 100, 2)
print(acc_gaussianage_1)
models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbkage_1, acc_sgdage_1, acc_knnage_1, acc_randomforestage_1, 
              acc_decisiontreeage_1, acc_perceptronage_1, acc_linear_svcage_1, acc_svcage_1, acc_logregage_1, 
               acc_gaussianage_1]})
models.sort_values(by='Score', ascending=False)
predictions = gbk.predict(x_tst_test_age.drop('PassengerId', axis=1))
p=0
for i in range(418):
    if np.isnan(test_test_age['AgeGroup'][i]) == True:
        test_test_age['AgeGroup'][i] = predictions[p]
        p+=1
from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.05, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
models = pd.DataFrame({
    'Model': ['Gradient Boosting Classifier','Stochastic Gradient Descent','KNN','Random Forest','Decision Tree', 'Perceptron','Linear SVC','Support Vector Machines', 
              'Logistic Regression','Naive Bayes',  
              ],
    'Score': [acc_gbk, acc_sgd, acc_knn, acc_randomforest, acc_decisiontree, acc_perceptron, acc_linear_svc, acc_svc, acc_logreg, 
               acc_gaussian]})
models.sort_values(by='Score', ascending=False)
#set ids as PassengerId and predict survival 
ids = test['PassengerId']
predictions = knn.predict(test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission_knn.csv', index=False)
x_tr = train.iloc[:,2:].as_matrix()
y_tr = train.iloc[:,1].as_matrix()
X_test = test.iloc[:,1:].as_matrix()
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(8,)))

model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu))

# model.add(tf.keras.layers.Flatten(input_shape=(8,)))

model.add(tf.keras.layers.Dense(2,activation = tf.nn.softmax))
model.summary()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
model.fit(x_tr, y_tr, batch_size=128, epochs = 60)
predictions = model.predict([X_test])
import csv
data = [['PassengerId', 'Survived']]
for i in range(1,419):
    data.append([i+891,np.argmax(predictions[i-1])])
print(data)
with open('submission_own_NN.csv','w',newline='') as fp:
    a=csv.writer(fp,delimiter = ',')
    a.writerows(data)
