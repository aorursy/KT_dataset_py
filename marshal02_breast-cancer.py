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

cols = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat']

# Any results you write to the current directory are saved as output.
import pandas_profiling as pp

import category_encoders as ce

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_selection import chi2

from xgboost import XGBClassifier

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn import preprocessing
data = (pd.read_csv("/kaggle/input/breast-cancer-data/breast-cancer.data", names = cols).replace({'?': 'unknown'}))
data = (pd.read_csv("/kaggle/input/breast-cancer-data/breast-cancer.data", names = cols).replace({'?': np.nan}))
data
pp.ProfileReport(data)
# Set up the matplotlib figure

fig, axes = plt.subplots(1,3, figsize=(15,6), sharex=False)

sns.countplot(x = data.columns[0], data = data, ax = axes[0])

sns.countplot(x = data.columns[2], data = data, ax = axes[1])

sns.countplot(x = data.columns[5], data = data, ax = axes[2])



fig, axes = plt.subplots(1,3, figsize=(15,6), sharex=False)

sns.countplot(x = data.columns[8], data = data, ax = axes[0])

sns.countplot(x = data.columns[9], data = data, ax = axes[1])

sns.countplot(x = data.columns[1], data = data, ax = axes[2])



fig, axes = plt.subplots(1,3, figsize=(15,6), sharex=False)

sns.countplot(x = data.columns[6], data = data, ax = axes[0])

sns.countplot(x = data.columns[7], data = data, ax = axes[1])

sns.countplot(x = data.columns[4], data = data, ax = axes[2])



fig, axes = plt.subplots(1,1, figsize=(15,6), sharex=False)

sns.countplot(x = data.columns[3], data = data, ax = axes)







X = data.drop('Class', axis = 1)

y = data['Class']
encode = ce.OneHotEncoder(handle_unknown='ignore', use_cat_names=True)

X = encode.fit_transform(X)
le = preprocessing.LabelEncoder()

y = le.fit_transform(y)
chi_scores = chi2(X,y)

p_values = pd.Series(chi_scores[1],index = X.columns)

p_values.sort_values(ascending = False , inplace = True)

p_values.plot.bar()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
classifier = KNeighborsClassifier(n_neighbors=6)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(accuracy_score(y_test, y_pred, normalize=True))
error = []



# Calculating error for K values between 1 and 40

for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))

    

plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')
'''

data = data.replace({'age': {'30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70-79': 5}, 

              'Class': {'no-recurrence-events': 0, 'recurrence-events': 1},

              'inv-nodes': {'0-2': 1, '3-5': 2, '6-8': 3, '9-11': 4, '12-14': 5, '15-17': 6, '24-26': 7},

              'tumor-size': {'0-4' : 1, '5-9': 2, '10-14': 3, '15-19': 4, '20-24': 5, '25-29': 6, '30-34': 7, '35-39': 8, '40-44': 9, '45-49': 10, '50-54': 11},

              'node-caps': {'no': 0, 'yes': 1, '?': np.nan},

              'breast': {'left': 0, 'right': 1},

              'irradiat': {'no': 0, 'yes': 1}})

'''
model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
X.shape
model = Sequential()

model.add(Dense(500, activation='relu', input_dim=41))

model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# Compile the model

model.compile(optimizer='adam', 

              loss='binary_crossentropy', 

              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
y_pred = model.predict_classes(X_test)

#predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))