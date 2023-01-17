# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

def warn(*args, **kwargs):

    pass

import warnings

warnings.warn = warn

import re

import os

print(os.listdir("../input"))



from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Dataset_spine.csv')

data.head()
Attribute = data.iloc[:,-1].dropna().tolist()

Attribute
column_name = []

for i in range(1, len(Attribute)-1):

    column_name.append(re.findall(r'\w+_\w+',Attribute[i])[0])

column_name.append('Class')

column_name
data.drop('Unnamed: 13', axis = 1, inplace = True)
data.columns = column_name
data.head()
data["Class"].value_counts().sort_index().plot.bar()
data = pd.get_dummies(data, drop_first=True)
data.describe()
X = data.iloc[:,:-1].values

y = data.iloc[:,-1:].values
std = StandardScaler()

X_scale = std.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2,random_state = 42)
modelLR1 = LogisticRegression(C = 0.5)

modelLR1.fit(X_train,y_train)

kfold = StratifiedKFold(n_splits = 10, shuffle=True , random_state=42)

results = cross_val_score(modelLR1, X, y, cv=kfold, scoring = 'accuracy')

print(results.mean())
train_sizes, train_scores, validation_scores = learning_curve(estimator=modelLR1,X= X_train,y = y_train, cv= 3,

                                                              scoring='accuracy')

train_scores_mean = train_scores.mean(axis = 1)

validation_scores_mean = validation_scores.mean(axis = 1)



plt.plot(train_sizes, train_scores_mean, label = 'Training accuracy')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation accuracy')

plt.ylabel('Accuracy', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

title = 'Learning curves' 

plt.title(title, fontsize = 18, y = 1.03)

plt.legend()
confusion_matrix(y_test, modelLR1.predict(X_test))
poly = PolynomialFeatures(degree = 2, interaction_only=True)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

X_train_poly.shape
modelLR2 = LogisticRegression(C = 0.5)

modelLR2.fit(X_train_poly,y_train)

kfold = StratifiedKFold(n_splits = 10, shuffle=True , random_state=42)

results = cross_val_score(modelLR2, X_train_poly, y_train, cv=kfold, scoring = 'accuracy')

print(results.mean())
train_sizes, train_scores, validation_scores = learning_curve(estimator=modelLR2,X= X_train_poly,y = y_train, cv= 3,

                                                              scoring='accuracy')

train_scores_mean = train_scores.mean(axis = 1)

validation_scores_mean = validation_scores.mean(axis = 1)



plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

title = 'Learning curves' 

plt.title(title, fontsize = 18, y = 1.03)

plt.legend()
confusion_matrix(y_test, modelLR2.predict(X_test_poly))
modelLR3 = LogisticRegression(C = 0.1)

modelLR3.fit(X_train_poly,y_train)

kfold = StratifiedKFold(n_splits = 10, shuffle=True , random_state=42)

results = cross_val_score(modelLR3, X, y, cv=kfold, scoring = 'accuracy')

print(results.mean())
train_sizes, train_scores, validation_scores = learning_curve(estimator=modelLR3,X= X_train_poly,y = y_train, cv= 3,

                                                              scoring='accuracy')

train_scores_mean = train_scores.mean(axis = 1)

validation_scores_mean = validation_scores.mean(axis = 1)



plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

title = 'Learning curves' 

plt.title(title, fontsize = 18, y = 1.03)

plt.legend()
confusion_matrix(y_test, modelLR3.predict(X_test_poly))
def create_model():

    model = Sequential()

    model.add(Dense(24, activation = 'relu', input_shape = (12,)))

#     model.add(Dropout(rate = 0.3))

    model.add(Dense(48, activation = 'relu'))

    model.add(Dropout(rate = 0.3))

    model.add(Dense(24, activation = 'relu'))

#     model.add(Dropout(rate = 0.3))

    model.add(Dense(6, activation = 'relu'))

#     model.add(Dropout(rate = 0.3))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model
model = create_model()

results = model.fit(X_train, y_train, batch_size=8, epochs = 100, validation_data=(X_test, y_test), verbose=0)

print(np.mean(results.history["val_acc"]))
modelann = KerasClassifier(build_fn=create_model, epochs = 50, batch_size = 8, verbose = 0)



kfold = StratifiedKFold(n_splits = 10, shuffle=True , random_state=42)

results = cross_val_score(modelann, X, y, cv=kfold, scoring = 'accuracy')

print(results.mean())
train_sizes, train_scores, validation_scores = learning_curve(estimator=modelann,X= X,y = y, cv= 3,

                                                              scoring='accuracy', random_state = 42)

train_scores_mean = train_scores.mean(axis = 1)

validation_scores_mean = validation_scores.mean(axis = 1)



plt.plot(train_sizes, train_scores_mean, label = 'Training error')

plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

plt.ylabel('Accuracy', fontsize = 14)

plt.xlabel('Training set size', fontsize = 14)

title = 'Learning curves' 

plt.title(title, fontsize = 18, y = 1.03)

plt.legend()