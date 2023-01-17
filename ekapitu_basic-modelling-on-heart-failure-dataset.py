# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
heart_data = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")

heart_data.describe()
heart_data.head()
# check for null values

heart_data.isnull().sum()
# Separate features and target variable

features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',

       'ejection_fraction', 'high_blood_pressure', 'platelets',

       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']



x = heart_data[features]

y = heart_data["DEATH_EVENT"]
import seaborn as sns

import matplotlib.pyplot as plt



plt.rcParams["figure.figsize"] = 12,1

plt.rcParams["font.size"] = 14



# Non boolean values

sns.boxplot(x=heart_data["creatinine_phosphokinase"], color="grey")

plt.show()

sns.boxplot(x=heart_data["ejection_fraction"], color="grey")

plt.show()

sns.boxplot(x=heart_data["platelets"], color="grey")

plt.show()

sns.boxplot(x=heart_data['serum_creatinine'], color = 'grey')

plt.show()

sns.boxplot(x=heart_data["serum_sodium"], color="grey")

plt.show()

sns.boxplot(x=heart_data["time"], color="grey")

plt.show()

sns.boxplot(x=heart_data["age"], color="grey")

plt.show()



# Boolean values: diabetes, sex, smoking, anaemia, high_blood_pressure
# Remove outliers

heart_data["ejection_fraction"] = heart_data[heart_data["ejection_fraction"] < 70]
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
# Normalize features

from sklearn.preprocessing import StandardScaler, LabelEncoder



scaler = StandardScaler().fit(x_train)

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)



encoder = LabelEncoder().fit(y)



y_train = encoder.fit_transform(y_train)

y_test = encoder.fit_transform(y_test)
# Storing the accuracy of all models

all_model_accuracy = {}
# Fit model

from sklearn.linear_model import LogisticRegression



model = LogisticRegression(random_state=0)

model.fit(x_train, y_train)



from sklearn.metrics import accuracy_score, confusion_matrix



# Predict, calculate score

y_pred = model.predict(x_test)



conf_matrix = confusion_matrix(y_pred, y_test)

accuracy = model.score(x_test, y_test)



all_model_accuracy["LogisticRegression"] = accuracy



print(conf_matrix)

print(accuracy)
# KNN model



from sklearn.neighbors import KNeighborsClassifier



# For grid search

accuracy_results = []



neighbor_range = range(3, 10)

for num_neighbors in neighbor_range:

    clf = KNeighborsClassifier(n_neighbors=num_neighbors, metric = 'minkowski')

    clf.fit(x_train, y_train)

#     y_pred = clf.predict(x_test_scaled)

    accuracy_results.append(clf.score(x_test, y_test))

    

import matplotlib.pyplot as plt



plt.plot(list(neighbor_range), accuracy_results)

plt.show()
# Using num neighbors = 9



knn_model = KNeighborsClassifier(n_neighbors=9, metric="minkowski")

knn_model.fit(x_train, y_train)



from sklearn.metrics import confusion_matrix, accuracy_score



y_pred = knn_model.predict(x_test)



conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)



accuracy = accuracy_score(y_test, y_pred)

all_model_accuracy["KNN"] = accuracy

print(accuracy)
from sklearn.tree import DecisionTreeClassifier



dtree_model = DecisionTreeClassifier(random_state=0, criterion="entropy")



dtree_model.fit(x_train, y_train)



y_predict = dtree_model.predict(x_test)



conf_matrix = confusion_matrix(y_pred, y_test)

accuracy = dtree_model.score(x_test, y_test)



all_model_accuracy["DecisionTree"] = accuracy



print(conf_matrix)

print(accuracy)
from sklearn.ensemble import RandomForestClassifier



accuracy_results = []

num_trees = range(100, 200)



for num_tree in num_trees:

    rf_model = RandomForestClassifier(random_state=0, n_estimators=num_tree, criterion='entropy')

    rf_model.fit(x_train, y_train)

    

    accuracy_results.append(rf_model.score(x_test, y_test))

    

import matplotlib.pyplot as plt



plt.plot(list(num_trees), accuracy_results)

plt.show()
# Using 110 trees

rf_model = RandomForestClassifier(n_estimators=110, random_state=0, criterion='entropy')

rf_model.fit(x_train, y_train)



y_pred = rf_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)



all_model_accuracy["RandomForest"] = accuracy

print(confusion_matrix(y_test, y_pred))

print(accuracy)
import tensorflow as tf

from tensorflow import keras

from keras import layers



ann_model = keras.Sequential()



# Add 4 layers

ann_model.add(layers.Dense(units=10, activation='relu'))

ann_model.add(layers.Dense(units=10, activation='relu'))

ann_model.add(layers.Dense(units=10, activation='relu'))

ann_model.add(layers.Dense(units=10, activation='relu'))



# Add output layer

ann_model.add(layers.Dense(units=1, activation='sigmoid'))



# Build

ann_model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'] )



# Train

ann_model.fit(x_train, y_train, batch_size = 32, epochs = 100)
# Summary of model

ann_model.summary()
# Predicting

y_predict = ann_model.predict(x_test)

y_predict = (y_predict > 0.5) # Convert to boolean

# y_predict
accuracy = accuracy_score(y_test, y_predict)



all_model_accuracy["NeuralNetwork"] = accuracy



print(confusion_matrix(y_predict, y_test))

print(accuracy_score(y_predict, y_test))
from xgboost import XGBClassifier



accuracy_results = []

num_estimators = range(10, 100, 10)



for num_estimator in num_estimators:

    model = XGBClassifier(n_estimators=num_estimator)

    model.fit(x_train, y_train)

    

    accuracy_results.append(model.score(x_test, y_test))

    

import matplotlib.pyplot as plt



plt.plot(list(num_estimators), accuracy_results)

plt.show()
model = XGBClassifier(n_estimators=80)

model.fit(x_train, y_train)



y_pred = model.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)



all_model_accuracy["XGBoost"] = accuracy



print(confusion_matrix(y_pred, y_test))

print(accuracy)
all_model_accuracy
import matplotlib.pylab as plt



font = {'family' : 'normal',

        'size'   : 13}



plt.rc('font', **font)



plt.figure(figsize=(15, 5))



ticks = range(1,7)

tick_label = list(all_model_accuracy.keys())

height = list(all_model_accuracy.values())



plt.bar(ticks, height, tick_label=tick_label)

plt.ylabel("Accuracy")

plt.xlabel("Classifier models")

plt.show()


