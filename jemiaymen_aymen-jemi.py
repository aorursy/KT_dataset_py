# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function, unicode_literals



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
test = pd.read_csv('/kaggle/input/dl50-project1/Test.csv')

train = pd.read_csv('/kaggle/input/dl50-project1/Train.csv')

sample = pd.read_csv('/kaggle/input/dl50-project1/sample.csv')
import matplotlib.pyplot as plt

from sklearn import preprocessing



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer



from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC



from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn import metrics

from sklearn.metrics import confusion_matrix,accuracy_score





import collections

import tensorflow as tf

from tensorflow.keras import layers



from sklearn.model_selection import StratifiedKFold

from keras.wrappers.scikit_learn import KerasClassifier
sample.head()
train.isna().sum()
train[train.isna().any(axis=1)]
train.duplicated().sum()
data = train

data.Month.unique()
data.VisitorType.unique()
def get_dummies(df,test = False):

    df.Month = df['Month'].map({'Feb' : 2, 'Mar' : 3, 'May' : 5, 'Oct': 10, 'June' : 6, 'Jul' : 7, 'Aug' : 8, 'Nov' : 11, 'Sep' : 9,'Dec' : 12}).astype(int)

    df.VisitorType = df['VisitorType'].map({'Returning_Visitor' : 2, 'New_Visitor' : 1, 'Other' : 3}).astype(int)

    df.Weekend = df['Weekend'].map( {True: 1, False: 0} ).astype(int)

    if test == False:

        df.Revenue = df['Revenue'].map( {True: 1, False: 0} ).astype(int)



get_dummies(data)

data.head()
data = data.drop(['id'], axis=1)
data.max()
def plot_df(data):

    plt.hist(data.Administrative_Duration,alpha=0.8, label='Administrative Duration')

    plt.hist(data.Informational_Duration, alpha=0.7, label='Informational Duration')

    plt.hist(data.ProductRelated_Duration, alpha=0.6, label='Product Related Duration')

    plt.hist(data.PageValues,alpha=0.5, label='Page Values')

    plt.legend(loc='upper right')

    plt.ylabel('Visitor')    
plot_df(data)
columns = ['Administrative_Duration' , 'Informational_Duration' , 'ProductRelated' , 'ProductRelated_Duration' , 'PageValues']

data_scaler = data[columns]

scaler = preprocessing.MinMaxScaler()

std_data = scaler.fit_transform(data_scaler)

std_data = pd.DataFrame(std_data,columns=columns)
plot_df(std_data)
data[columns] = std_data
imp = IterativeImputer(random_state=0)

data_clean = imp.fit_transform(data)

data_clean = pd.DataFrame(data_clean , columns =  ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues','SpecialDay','Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue'])

data_clean[data_clean.isna().any(axis=1)]
X = data_clean.drop(['Revenue'], axis=1)

y = data_clean[['Revenue']]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
model = tf.keras.Sequential()

model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal', input_dim=17))

model.add(layers.Dense(256, activation='relu' , kernel_initializer='random_normal'))

model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal'))

model.add(layers.Dense(1, activation='sigmoid'))

model.summary() 

model.compile(loss='binary_crossentropy', optimizer='Nadam',metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=(X_test, y_test),batch_size=200,epochs=20)

eval_model=model.evaluate(X_train, y_train)

eval_model
clstm = tf.keras.Sequential()

clstm.add(layers.Embedding(300, 300, input_length=17))

clstm.add(layers.LSTM(200))

clstm.add(layers.Dense(1, activation='sigmoid'))

clstm.summary() 

clstm.compile(loss='binary_crossentropy', optimizer='Nadam',metrics=['accuracy'])
clstm.fit(X_train, y_train,validation_data=(X_test, y_test),batch_size=64,epochs=5)

eval_model=clstm.evaluate(X_train, y_train)

eval_model
dtree = DecisionTreeClassifier()

dtree = dtree.fit(X_train, y_train)



y_pred = dtree.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

accuracy
rforest = RandomForestClassifier()

rforest = rforest.fit(X_train, y_train)



y_pred = rforest.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

accuracy
k_scores = []

for i in range(1,25) :   

    knn = KNeighborsClassifier(n_neighbors = i)

    knn = knn.fit(X_train,y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)

    k_scores.append(accuracy)

np.mean(k_scores)
svm = SVC()

svm = svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

accuracy
k_fold = KFold(n_splits = 10, shuffle = True, random_state  =0)

scoring = 'accuracy'
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

cvscores = []

_X = X.values

_y = y.values



for train, test in kfold.split(_X,_y):

    

    nn_best = tf.keras.Sequential()

    nn_best.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal', input_dim=17))

    nn_best.add(layers.Dense(256, activation='relu' , kernel_initializer='random_normal'))

    nn_best.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal'))

    nn_best.add(layers.Dense(1, activation='sigmoid'))

    nn_best.compile(loss='binary_crossentropy', optimizer='Nadam',metrics=['accuracy'])

    nn_best.fit(_X[train], _y[train],validation_data=(_X[test], _y[test]),batch_size=200,epochs=20,verbose=0)

    scores=nn_best.evaluate(_X[test], _y[test])

    print("%s: %.2f%%" % (nn_best.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)



    

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))







cvscores
def create_model(optimizer='adam'):

    model = tf.keras.Sequential()

    model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal', input_dim=17))

    model.add(layers.Dense(256, activation='relu' , kernel_initializer='random_normal'))

    model.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model



model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=200, verbose=0)

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_grid = dict(optimizer=optimizer)

grid = GridSearchCV(estimator=model, param_grid=param_grid,  cv=3)

grid_result = grid.fit(X, y)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
tree = DecisionTreeClassifier()

score = cross_val_score(tree, X, y, cv= k_fold, n_jobs=1, scoring=scoring)

score
rforest = RandomForestClassifier()

score = cross_val_score(rforest, X, y, cv= k_fold, n_jobs=1, scoring=scoring)

score
k_scores = []

for i in range(10,50) :   

    knn = KNeighborsClassifier(n_neighbors = i)

    score = cross_val_score(knn, X, y, cv = k_fold, scoring = scoring)  

    k_scores.append(np.mean(score))

np.mean(k_scores)
k_range = range(10, 50)

plt.plot(k_range, k_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross-Validated Accuracy')
svm = SVC()

score = cross_val_score(svm, X, y, cv= k_fold, n_jobs=1, scoring=scoring)

score
param_grid = {'n_neighbors': np.arange(10,20)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, param_grid, cv=10,scoring = scoring)

knn_cv.fit(X, y)

knn_cv.best_params_
knn_best = KNeighborsClassifier(n_neighbors = 18)

score = cross_val_score(knn_best, X, y, cv = k_fold, n_jobs = 1, scoring = scoring)

score
# param_grid = {'kernel': ('linear', 'rbf','poly'),'gamma': [0.001, 0.01, 0.1, 1],'C':[0.001, 0.01, 0.1, 1, 10]}

# svm = SVC()

# svm_cv = GridSearchCV(svm, param_grid, cv=10)

# svm_cv.fit(X, y) 

# svm_cv.best_params_
param_grid = {"max_depth": [3, None],"max_features": [1, 3, 10],"min_samples_split": [2, 3, 10],"bootstrap": [True, False],"criterion": ["gini", "entropy"]}

rforest = RandomForestClassifier()

forest_cv = GridSearchCV(rforest,param_grid,cv=10)

forest_cv = forest_cv.fit(X,y)

forest_cv.best_params_
forest_best = RandomForestClassifier(bootstrap = True,criterion = 'gini',max_depth = 3,max_features =  10,min_samples_split = 10)

score = cross_val_score(forest_best, X, y, cv = k_fold, n_jobs = 1, scoring = scoring)

np.mean(score)
depths = np.arange(1, 21)

num_leafs = [1, 5, 10, 20, 50, 100]

param_grid = [{'max_depth':depths,'min_samples_leaf':num_leafs}]

tree = DecisionTreeClassifier()

tree_cv = GridSearchCV(tree, param_grid, cv=10)

tree_cv = tree_cv.fit(X,y)

tree_cv.best_params_
tree_best = DecisionTreeClassifier(max_depth = 4 , min_samples_leaf = 50)

score = cross_val_score(tree_best, X, y, cv = k_fold, n_jobs = 1, scoring = scoring)

score

np.mean(score)
test = pd.read_csv('/kaggle/input/dl50-project1/Test.csv')

data_test = test



data_test = data_test.drop(['id'], axis=1)



get_dummies(data_test,test = True)

data_test.head()
columns = ['Administrative_Duration' , 'Informational_Duration' , 'ProductRelated' , 'ProductRelated_Duration' , 'PageValues']

data_scaler = data_test[columns]

scaler = preprocessing.MinMaxScaler()

test_std_data = scaler.fit_transform(data_scaler)

test_std_data = pd.DataFrame(test_std_data,columns=columns)

data_test[columns] = test_std_data

data_test.head()
tree_best.fit(X, y)

prediction = tree_best.predict(data_test)

submission = test[['id']]

submission['Revenue'] = prediction

#submission.Revenue = submission['Revenue'].map( {1: True, 0 : False} ).astype(bool)

submission.to_csv("submission_tree.csv", index = False)
forest_best.fit(X, y)

prediction = forest_best.predict(data_test)



submission = test[['id']]

submission['Revenue'] = prediction

#submission.Revenue = submission['Revenue'].map( {1: True, 0 : False} ).astype(bool)

submission.to_csv("submission_forest.csv", index = False)
nn_best = tf.keras.Sequential()

nn_best.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal', input_dim=17))

nn_best.add(layers.Dense(256, activation='relu' , kernel_initializer='random_normal'))

nn_best.add(layers.Dense(128, activation='relu' , kernel_initializer='random_normal'))

nn_best.add(layers.Dense(1, activation='sigmoid'))

nn_best.compile(loss='binary_crossentropy', optimizer='Nadam',metrics=['accuracy'])

nn_best.fit(X,y,batch_size=200,epochs=20,verbose=0)

    

nn_best.fit(X, y)

prediction = nn_best.predict(data_test)

submission = test[['id']]

submission['Revenue'] = prediction

#submission.Revenue = submission['Revenue'].map( {1: True, 0 : False} ).astype(bool)

submission.to_csv("submission_nn.csv", index = False)