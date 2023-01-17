import os

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix , accuracy_score ,mean_squared_error

from math import sqrt
data = pd.read_csv("../input/heart-disease-uci/heart.csv")

data.info()



data = data.drop('oldpeak', axis = 1) 
data.info()

data.info()

print (data)

t = data.head(10)

print (t)

data.shape

data.describe()
from sklearn.model_selection import train_test_split

X = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal']]

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

X_train

X_test

y_train

y_test 

print (X_train)

print (X_test)

print (y_train)

print (y_test)
data['goal'] = 'na'

data1= data.copy()

data2= data.copy()

data3= data.copy()

data1.loc[(data1.target <= 0) , 'goal'] = 'Injured'

data1.loc[(data1.target > 0) , 'goal'] = 'unInjured'

data1 = data1.drop([ 'target'], axis=1)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

data2.target = le.fit_transform(data2.target)
X = data2.drop('target',axis=1)

y = data2.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)

len(list(X_train))
from sklearn.tree import DecisionTreeClassifier , DecisionTreeRegressor 

from sklearn.neighbors import KNeighborsClassifier as knModel

from sklearn.neural_network import MLPClassifier , MLPRegressor

from sklearn.naive_bayes import GaussianNB as naive_classifier1
from sklearn.naive_bayes import GaussianNB

# model = GaussianNB()

from sklearn.metrics import accuracy_score



from sklearn.tree import DecisionTreeClassifier as dtc

model = dtc()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, predictions)

print('Accuracy: ',accuracy)
confMatrix = metrics.confusion_matrix(y_test, predictions)

print(confMatrix)
classReport = metrics.classification_report(y_test, predictions)

print(classReport)
msl=[]

for i in range(1,20):

    tree = DecisionTreeClassifier(min_samples_leaf=i)

    t= tree.fit(X_train, y_train)

    ts=t.score(X_test, y_test)

    msl.append(ts)

msl = pd.Series(msl)

msl.where(msl==msl.max()).dropna()

def evaluate_classification(X,y):

    le = preprocessing.LabelEncoder()

    y = le.fit_transform(y)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    # get dummy varibles - Convert categorical variable into dummy/indicator variables.

    X_train = pd.get_dummies(X_train)

    X_test = pd.get_dummies(X_test)

    

    classifier = ['DecisionTreeClassifier', 'KNeighborsClassifier','GaussianNB']

    

    # Instantiate the models   

    classifier1 = DecisionTreeClassifier(max_leaf_nodes=17 ,random_state=100)

    classifier2 = knModel(n_neighbors=5 , metric='minkowski' , p=2)

    classifier3 =  naive_classifier1()

    

    

    # Dataframe for results

    results = pd.DataFrame(columns=['accuracy'], index=classifier)



    # Train and predict with each model

    for i, classifier1 in enumerate([classifier1, classifier2, classifier3,classifier4]):

        classifier1.fit(X_train, y_train)

        predictions = classifier1.predict(X_test)

        

        # Metrics

        classr = classifier[i]

        print('Result For ' , classr )

        acc = accuracy_score(y_test, predictions)

        confusion = confusion_matrix(y_test, predictions)

        report = classification_report(y_test,predictions)



        print("Accuracy:", acc)

        print("Confusion Matrix:\n", confusion)

        print("Classification Report:\n",report)



        

        # Metrics

        classr = classifier[i]

        acc = accuracy_score(y_test, predictions)

        # Insert results into the dataframe

        classr =classifier[i]

        results.loc[classr, :] = [acc]

    return results

def evaluate_regression(X,y):

    le = preprocessing.LabelEncoder()

    y = le.fit_transform(y)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    # get dummy varibles - Convert categorical variable into dummy/indicator variables.

    X_train = pd.get_dummies(X_train)

    X_test = pd.get_dummies(X_test)

    

    model_name_list = ['MLPClassifier' , 'RandomForestClassifier' , 'DecisionTreeClassifier']

    

    # Instantiate the models

    model1 = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

    model2 = DecisionTreeRegressor(min_samples_leaf=17)

    

    # Dataframe for results

    results = pd.DataFrame(columns=['RMSE'], index= classifier)



    # Train and predict with each model

    for i, model in enumerate([classifier1, classifier2]):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Metrics

        model_name = classifier[i]

        rmse = sqrt(mean_squared_error(y_test, predictions))



        # Insert results into the dataframe

        model_name =classifier[i]

        results.loc[model_name, :] = [rmse]

    return results
def evaluate_regression(X,y):

    le = preprocessing.LabelEncoder()

    y = le.fit_transform(y)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

    # get dummy varibles - Convert categorical variable into dummy/indicator variables.

    X_train = pd.get_dummies(X_train)

    X_test = pd.get_dummies(X_test)

    

    model_name_list = ['MLPClassifier' , 'RandomForestClassifier' , 'DecisionTreeClassifier']

    

    # Instantiate the models

    model1 = MLPRegressor(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,solver='lbfgs')

    model2 = DecisionTreeRegressor(min_samples_leaf=17)

    

    # Dataframe for results

    results = pd.DataFrame(columns=['RMSE'], index= classifier)



    # Train and predict with each model

    for i, model in enumerate([classifier1, classifier2]):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Metrics

        model_name = classifier[i]

        rmse = sqrt(mean_squared_error(y_test, predictions))



        # Insert results into the dataframe

        model_name =classifier[i]

        results.loc[model_name, :] = [rmse]

    return results
cont_data=data.copy()
results = pd.DataFrame(columns=[ 'MLPClassifier' , 'RandomForestClassifier' , 'DecisionTreeClassifier'], index=['A' , 'B' , 'C'])

m = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal']]

s = data['target']
accuracy = metrics.accuracy_score(y_test, predictions)

results.loc['A']=  accuracy



results.loc['B'] =  accuracy 

results.loc['C'] = accuracy 



results

classReport

