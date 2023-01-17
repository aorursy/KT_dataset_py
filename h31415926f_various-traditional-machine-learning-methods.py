# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import metrics

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/adult.csv')

# Delete rows with missing data

data = data.loc[(data['workclass']!='?') & (data['occupation']!='?') & (data['native.country']!='?')]
def mapping(column_str):

    data_ = data[column_str]

    list_ = list(set(data_))

    map_ = {}

    for i in range(len(list_)):

        map_[list_[i]] = i

    return data_.map(map_)



# Mapping non-numeric data to numeric data

non_numeric = ["workclass", "marital.status", "occupation", "relationship", "race", "sex", "native.country", "income"]

for column in non_numeric:

    data[column] = mapping(column)
X = data[['age','workclass', 'education.num', 'marital.status','occupation', 'relationship', 'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country']]

Y = data["income"]
# split data to train and test

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=2019)
#LR

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(train_X, train_Y)

pred_Y = lr.predict(test_X)

print('LR:')

print('accuracy %s' % metrics.accuracy_score(test_Y, pred_Y))

print('F1_score %s' % metrics.f1_score(test_Y, pred_Y))

print('auc %s' % metrics.roc_auc_score(test_Y, pred_Y))
# KNN

from sklearn.neighbors import KNeighborsClassifier

k_range = np.arange(1, 31)

accuracy_scores = []

F1_scores = []

auc_scores = []

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(train_X, train_Y)

    pred_Y = knn.predict(test_X)

    accuracy_scores.append(metrics.accuracy_score(test_Y, pred_Y))

    F1_scores.append(metrics.f1_score(test_Y, pred_Y))

    auc_scores.append(metrics.roc_auc_score(test_Y, pred_Y))

print('KNN:')

print('accuracy : %s' % accuracy_scores.index(max(accuracy_scores)), max(accuracy_scores))

print('F1_score : %s' % F1_scores.index(max(F1_scores)), max(F1_scores))

print('auc : %s' % auc_scores.index(max(auc_scores)), max(auc_scores))
#plot k and metrics in knn model

#accuracy

plt.figure(1)

plt.plot(k_range, accuracy_scores)

plt.xlabel('K')

plt.ylabel('Accuracy')

#F1_score

plt.figure(2)

plt.plot(k_range, F1_scores)

plt.xlabel('K')

plt.ylabel('F1_scores')

#auc

plt.figure(3)

plt.plot(k_range, auc_scores)

plt.xlabel('K')

plt.ylabel('auc')
# Decision Tree

from sklearn import tree

deep_range = np.arange(1,21)

accuracy_scores = []

F1_scores = []

auc_scores = []

for max_deep in deep_range:

    dt = tree.DecisionTreeClassifier(max_depth=max_deep)

    dt = dt.fit(train_X, train_Y)

    pred_Y = dt.predict(test_X)

    accuracy_scores.append(metrics.accuracy_score(test_Y, pred_Y))

    F1_scores.append(metrics.f1_score(test_Y, pred_Y))

    auc_scores.append(metrics.roc_auc_score(test_Y, pred_Y))

print('Decision Tree:')

print('accuracy : %s' % accuracy_scores.index(max(accuracy_scores)), max(accuracy_scores))

print('F1_score : %s' % F1_scores.index(max(F1_scores)), max(F1_scores))

print('auc : %s' % auc_scores.index(max(auc_scores)), max(auc_scores))
#plot max_depth in decision tree model and metrics

#accuracy

plt.figure(1)

plt.plot(deep_range, accuracy_scores)

plt.xlabel('max_depth')

plt.ylabel('Accuracy')

#F1_score

plt.figure(2)

plt.plot(deep_range, F1_scores)

plt.xlabel('max_depth')

plt.ylabel('F1_scores')

#auc

plt.figure(3)

plt.plot(deep_range, auc_scores)

plt.xlabel('max_depth')

plt.ylabel('auc')
#NB

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB().fit(train_X, train_Y)

pred_Y = nb.predict(test_X)

print('NB:')

print('accuracy %s' % metrics.accuracy_score(test_Y, pred_Y))

print('F1_score %s' % metrics.f1_score(test_Y, pred_Y))

print('auc %s' % metrics.roc_auc_score(test_Y, pred_Y))
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=80, max_features='auto', max_depth=15)

rf = rf.fit(train_X, train_Y)

pred_Y = rf.predict(test_X)

print('Random Forest:')

print('accuracy %s' % metrics.accuracy_score(test_Y, pred_Y))

print('F1_score %s' % metrics.f1_score(test_Y, pred_Y))

print('auc %s' % metrics.roc_auc_score(test_Y, pred_Y))
#GB

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, max_depth=4)

gbc = gbc.fit(train_X, train_Y)

pred_Y = gbc.predict(test_X)

print('GBC:')

print('accuracy %s' % metrics.accuracy_score(test_Y, pred_Y))

print('F1_score %s' % metrics.f1_score(test_Y, pred_Y))

print('auc %s' % metrics.roc_auc_score(test_Y, pred_Y))
#SVM

from sklearn.svm import SVC

svm = SVC()

svm = svm.fit(train_X, train_Y)

pred_Y = svm.predict(test_X)

print('SVM:')

print('accuracy %s' % metrics.accuracy_score(test_Y, pred_Y))

print('F1_score %s' % metrics.f1_score(test_Y, pred_Y))

print('auc %s' % metrics.roc_auc_score(test_Y, pred_Y))
#Neural Networks

from keras.models import Sequential

from keras.layers import Dense,Dropout,BatchNormalization

model=Sequential([

    BatchNormalization(input_shape=(12,)),

    Dense(16,activation='relu'),

    Dropout(0.5),

    Dense(8,activation='relu'),

    Dropout(0.5),

    Dense(4,activation='relu'),

    Dense(1,activation='sigmoid'),

])

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x=train_X, y=train_Y, batch_size=256, epochs=30)
y_pred = np.squeeze(model.predict_classes(test_X))

print('Nerual Network:')

print('accuracy %s' % metrics.accuracy_score(test_Y, pred_Y))

print('F1_score %s' % metrics.f1_score(test_Y, pred_Y))

print('auc %s' % metrics.roc_auc_score(test_Y, pred_Y))