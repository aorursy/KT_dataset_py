import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import math



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train['Title'] = train['Name'].apply(lambda x : x.split(' '))

train['Title'] = train['Title'].apply(lambda x: [each if '.' in each else None for each in x])

train['Title'] = train['Title'].apply(lambda x: [each for each in x if each][0])





test['Title'] = test['Name'].apply(lambda x : x.split(' '))

test['Title'] = test['Title'].apply(lambda x: [each if '.' in each else None for each in x])

test['Title'] = test['Title'].apply(lambda x: [each for each in x if each][0])
train['Ticket_Type'] = train['Ticket'].apply(lambda x: "".join(x.split(' ')[:-1]) if len(x.split(' '))>1 else 'Normal')

train['Ticket_Type'] = train['Ticket_Type'].apply(lambda x:x.replace('.','').replace('/',''))



test['Ticket_Type'] = test['Ticket'].apply(lambda x: "".join(x.split(' ')[:-1]) if len(x.split(' '))>1 else 'Normal')

test['Ticket_Type'] = test['Ticket_Type'].apply(lambda x:x.replace('.','').replace('/',''))
combined = pd.concat([train,test])
age_mean_dict = combined.groupby(['Title'])['Age'].mean().to_dict()



new_age_col_train=[]

for i in range(len(train)):

    if math.isnan(train.loc[i]['Age']):

        new_age_col_train.append(age_mean_dict[train.loc[i]['Title']])

    else:

        new_age_col_train.append(train.loc[i]['Age'])

train['Age'] = new_age_col_train



new_age_col_test=[]

for i in range(len(test)):

    if math.isnan(test.loc[i]['Age']):

        new_age_col_test.append(age_mean_dict[test.loc[i]['Title']])

    else:

        new_age_col_test.append(test.loc[i]['Age'])

test['Age'] = new_age_col_test
train['Embarked'] = train['Embarked'].fillna('S')

test['Embarked'] = test['Embarked'].fillna('S')
feature_list = ['Sex','Embarked','Age','Pclass','SibSp','Parch']



#Adding Column in Feature Vector

X_train = train[feature_list]

X_test = test[feature_list]



#Creating 'Y', Output Lable for training set

Y_train = np.array(train[['Survived']]).reshape(len(X_train),)
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')

X_train = np.array(ct.fit_transform(X_train))

X_test = np.array(ct.fit_transform(X_test))



ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [2])], remainder='passthrough')

X_train = np.array(ct.fit_transform(X_train))

X_test = np.array(ct.fit_transform(X_test))
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.svm import SVC



classifier = SVC(kernel = 'rbf', random_state = 42)

classifier.fit(X_train, Y_train)



Y_train_hat = classifier.predict(X_train)



Y_test_hat = classifier.predict(X_test)
from sklearn.ensemble import RandomForestClassifier

classifier_2 = RandomForestClassifier(n_estimators = 100,max_depth=5,criterion='gini')

classifier_2.fit(X_train, Y_train)



Y_train_hat = classifier_2.predict(X_train)



Y_test_hat = classifier_2.predict(X_test)
from xgboost import XGBClassifier

classifier_3 = XGBClassifier()

classifier_3.fit(X_train, Y_train)



Y_train_hat = classifier_3.predict(X_train)



Y_test_hat = classifier_3.predict(X_test)
#Defining Model and Training

import tensorflow as tf

ann = tf.keras.models.Sequential()

ann.add(tf.keras.layers.Dense(units=X_train.shape[1], activation='relu'))

ann.add(tf.keras.layers.Dense(units=X_train.shape[1], activation='relu'))

ann.add(tf.keras.layers.Dense(units=X_train.shape[1], activation='relu'))

ann.add(tf.keras.layers.Dense(units=int(round(X_train.shape[1]/2,0)), activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))





ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, Y_train, epochs = 100)
#Testing the Model

Y_train_hat = ann.predict(X_train).reshape(len(X_train),1)

Y_train_hat = [0 if each<=0.5 else 1 for each in Y_train_hat]



Y_test_hat = ann.predict(X_test).reshape(len(X_test),1)

Y_test_hat = [0 if each<=0.5 else 1 for each in Y_test_hat]
model_to_validate = classifier
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_train_hat, Y_train)



print('Prediction Accuracy on the Training Set : ',(cm[0,0]+cm[1,1])*100/sum(sum(cm)))
from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator = model_to_validate, X = X_train, y = Y_train, cv = 10)



print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
from sklearn.model_selection import GridSearchCV



# Parameter list for various classification models

# Choose accoring to the model set in 'model_to_validate'

parameters_SVC = [{'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.3, 0.5, 0.7, 0.9]}]



parameters_RF = [{'n_estimators': [10,50,100,150,200,250,300], 'max_depth': [4,5,6,7,8,9,10], 'criterion':['gini','entropy']}]



grid_search = GridSearchCV(estimator = model_to_validate,

                           param_grid = parameters_SVC,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)



grid_search = grid_search.fit(X_train, Y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_



print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)