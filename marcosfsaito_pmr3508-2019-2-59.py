import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

import os
cwd = os.getcwd()

directory = '../input/adult-pmr3508'

files = os.listdir(directory)
try: 

    data = pd.read_csv(directory+'/train_data.csv',names = ['ID','Age','fnlwgt','Workclass','Education','Education Num','Marital Status',

                        'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',

                        'Hours per Week','Native Country','Income'],skiprows=1,sep=',',na_values='?')

except FileNotFoundError as error:

    print('No train data')

    

data = data.iloc[:,1:]

data = data.drop(columns = 'Education'); # education level is expressed in education.num

                                         # as a numeric feature

data = data.drop(columns = 'fnlwgt');     # no significant data
data = data.dropna()
sex = data.loc[:,'Sex']

race = data.loc[:,'Race']

income = data.loc[:,'Income']

incomexsex = pd.crosstab(income,race,margins=False,normalize='columns')

#incomexsex.plot(kind='bar',stacked=False);
from sklearn import preprocessing as prep



data = data.apply(prep.LabelEncoder().fit_transform)
X = data.iloc[:,0:-1];

Y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components=1)

x_train = lda.fit_transform(x_train, y_train)

x_test = lda.transform(x_test)
from sklearn.neighbors import KNeighborsClassifier as knnClassifier

from sklearn.model_selection import cross_val_score
results=[0]*30

for i in range(30):

    knn = knnClassifier(n_neighbors = 32+i)

    scores = cross_val_score(knn, x_train, y_train,cv=10)

    results[i] = np.mean(scores)
n = np.argmax(results)

knn = knnClassifier(n_neighbors = 50)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

#knn.score(x_test,y_test)



#Accuracy score of KNN

from sklearn.metrics import accuracy_score

knn_score = accuracy_score(y_test,y_pred)
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees

rf = RandomForestClassifier(n_estimators = 500, random_state = True)
# Train the model on training data

rf.fit(x_train, y_train)
# Use the forest's predict method on the test data

y_pred = rf.predict(x_test)

rf_score = accuracy_score(y_test,y_pred)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='newton-cg')

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

lr_score = accuracy_score(y_test,y_pred)
from sklearn.svm import SVC

svm = SVC(C=2, gamma='scale')

svm.fit(x_train,y_train)

y_pred = svm.predict(x_test)

svm_score = accuracy_score(y_test,y_pred)
from sklearn.neural_network import MLPClassifier as mlp

neuralnet = mlp(hidden_layer_sizes=(100,))

neuralnet.fit(x_train,y_train)

y_pred = neuralnet.predict(x_test)

neuralnet_score = accuracy_score(y_test,y_pred)
from sklearn.ensemble import AdaBoostClassifier as ada

from sklearn.tree import DecisionTreeClassifier

boost = ada(base_estimator=DecisionTreeClassifier(max_depth=1))

boost.fit(x_train,y_train)

y_pred_boost = boost.predict(x_test)

boost_score = accuracy_score(y_test,y_pred_boost)
boost2 = ada(base_estimator=RandomForestClassifier(

    n_estimators = 10, random_state = True))

boost2.fit(x_train,y_train)

y_pred = boost2.predict(x_test)

boost2_score = accuracy_score(y_test,y_pred)
accuracies = pd.DataFrame([knn_score,rf_score,lr_score,svm_score,neuralnet_score,

                          boost_score,boost2_score],

                          ['KNN','Random Forest','Logistic Regression','SVM',

                           'Neural Network','ADABoost with Decision Tree',

                           'ADABoost with Random Forest'])

accuracies.columns=['Accuracy Scores']
accuracies
income_pred = pd.DataFrame(y_pred_boost)

income_pred.to_csv("submission.csv",header = ["income"], index_label = "Id")