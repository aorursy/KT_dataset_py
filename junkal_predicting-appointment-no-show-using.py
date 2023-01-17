import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

import time
data = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')

#data.head(50)
print(data.shape)
#Print total number of NaNs

data.isnull().sum()
gender_map = {'M': 1, 'F': 2}

DayOfTheWeek_map = {'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}

Status_map = {'Show-Up': 1, 'No-Show': 0}



data['Gender'] = data.Gender.map(gender_map)

data['DayOfTheWeek'] = data.DayOfTheWeek.map(DayOfTheWeek_map)

data['Status'] = data.Status.map(Status_map)
data = data.rename(columns = {'Alcoolism':'Alchoholism', 

                              'HiperTension' : "Hypertension", 

                              'Handcap':'Handicap', 

                              'ApointmentData':'AppointmentDate'})
data.AppointmentRegistration = data.AppointmentRegistration.apply(np.datetime64)

data.AppointmentDate = data.AppointmentDate.apply(np.datetime64)



data['AppointmentRegistration-Date'] = pd.to_datetime(data.AppointmentRegistration).dt.date

data['AppointmentRegistration-Time'] = pd.to_datetime(data.AppointmentRegistration).dt.time



data['AppointmentDate-Date'] = pd.to_datetime(data.AppointmentDate).dt.date

data['AppointmentDate-Time'] = pd.to_datetime(data.AppointmentDate).dt.time



data = data.drop('AppointmentRegistration', axis=1)

data = data.drop('AppointmentDate', axis=1)



data.head(10)
data.drop('AppointmentDate-Time', axis=1, inplace=True)

data.drop('AppointmentDate-Date', axis=1, inplace=True)

data.drop('AppointmentRegistration-Date', axis=1, inplace=True)

data.drop('AppointmentRegistration-Time', axis=1, inplace=True) 
print(data.groupby('Status').size())
data.to_csv('No-show-Issue-Comma-300k_cleaned.csv', index=False)
models_list = []

models_list.append(('CART', DecisionTreeClassifier()))

models_list.append(('Linear SVM', LinearSVC())) 

models_list.append(('NB', GaussianNB()))

models_list.append(('KNN', KNeighborsClassifier()))
data = pd.read_csv('No-show-Issue-Comma-300k_cleaned.csv')

sample_size = 10000

data_sample = data.sample(n=sample_size, replace=False, random_state= 123)



# Create arrays for the features and the response variable

Y_sample = data_sample['Status'].values

X_sample = data_sample.drop('Status', axis=1).values
num_folds = 10

results = []

names = []



for name, model in models_list:

    kfold = KFold(n_splits=num_folds, random_state=123)

    start = time.time()

    cv_results = cross_val_score(model, X_sample, Y_sample, cv=kfold, scoring='accuracy')

    end = time.time()

    results.append(cv_results)

    names.append(name)

    print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))

fig = plt.figure()

fig.suptitle('Performance Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
import warnings



# Standardize the dataset

pipelines = []



pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',

                                                                        DecisionTreeClassifier())])))

pipelines.append(('ScaledLinearSVM', Pipeline([('Scaler', StandardScaler()),('Linear SVM', LinearSVC( ))])))

pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',

                                                                      GaussianNB())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',

                                                                       KNeighborsClassifier())])))

results = []

names = []

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    kfold = KFold(n_splits=num_folds, random_state=123)

    for name, model in pipelines:

        start = time.time()

        cv_results = cross_val_score(model, X_sample, Y_sample, cv=kfold, scoring='accuracy')

        end = time.time()

        results.append(cv_results)

        names.append(name)

        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))
fig = plt.figure()

fig.suptitle('Performance Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
scaler = StandardScaler().fit(X_sample)

X_sample_train_scaled = scaler.transform(X_sample)

c_values = [0.1, 0.5, 1.0, 1.5, 2.0]

kfold = KFold(n_splits=num_folds, random_state=123)

overall_mean = []

overall_std = []



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    for c_val in c_values:

        model = LinearSVC (C=c_val)

        start = time.time()

        cv_results = cross_val_score(model, X_sample_train_scaled, Y_sample, cv=kfold, scoring='accuracy')

        end = time.time()

        results.append(cv_results)

        names.append(name)

        print( "Linear SVC (C=%f): %f (%f) (run time: %f)" % (c_val, cv_results.mean(), cv_results.std(), end-start))

        overall_mean.append(cv_results.mean())

        overall_std.append(cv_results.std())

# Create arrays for the features and the response variable

Y = data['Status'].values

X = data.drop('Status', axis=1).values
# Create training and test sets

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.30, random_state=21)
# prepare the model

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

model = LinearSVC(C=1.0)

start = time.time()

model.fit(X_train_scaled, Y_train)

end = time.time()

print( "Run Time: %f" % (end-start))
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# estimate accuracy on test dataset

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)

print("Accuracy score %f" % accuracy_score(Y_test, predictions))

print(classification_report(Y_test, predictions))
from nltk import ConfusionMatrix

print(ConfusionMatrix(list(Y_test), list(predictions)))
model = GaussianNB()

model.fit(X_train, Y_train)

predictions = model.predict(X_test_scaled)

print("Accuracy score %f" % accuracy_score(Y_test, predictions))

print(classification_report(Y_test, predictions))
print(ConfusionMatrix(list(Y_test), list(predictions)))