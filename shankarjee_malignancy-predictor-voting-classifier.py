# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pylab as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/data.csv')
data.head()
# Check to see if there is any null data
data.isnull().sum()
# We can drop id, Unnamed: 32 is only Nan. we will drop that also
data.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)
# The diagnosis is M or B. Let us convert it to categorical data
data.diagnosis = pd.Categorical( data.diagnosis ).codes #M - 1, B - 0
from pandas.plotting import scatter_matrix
def getCorrColumns(data, id_x = 0, toler = 0.97):
    cm = data.corr()    
    idx = cm.iloc[id_x] > toler    
    drop_cols = data.columns[idx].values[1:]
    if drop_cols.size > 0:
        scatter_matrix( data.loc[:,idx], figsize = (10,10), alpha = 0.2, diagonal = 'kde')
        plt.suptitle('Correlated values for ' + data.columns[id_x], fontsize = 25)
        plt.show()
        print('Dropping ' , drop_cols)
        data.drop(drop_cols, axis = 1, inplace = True)    
X = data.drop('diagnosis', axis = 1)
idx = 0
while True:
    getCorrColumns(X, idx)
    idx += 1
    if idx == X.columns.size: break

print('*'*50)
print('Initial columns: ', data.columns.size)
print('Final columns  : ', X.columns.size)
print('*'*50)
fig, ax = plt.subplots(1,2)
ax[0].plot( data.area_mean/data.radius_mean**2)
ax[0].plot( [0,568], [3.08, 3.08], c='r', ls=':', lw = 3)
ax[0].set_title('Area to Radius')

ax[1].plot( data.perimeter_mean/data.radius_mean)
ax[1].plot( [0,568], [6.55]*2, c='r', ls=':', lw = 3)
ax[1].set_title('Perimeter to Radius')

plt.show()

# Our hypothesis seems to be fine
# First let us scale the data
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

X_sc = scaler.fit_transform(X)
y = data.diagnosis.values

# Split into test and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_sc, y, test_size = 0.2)
# Let us choose some fitting algorithms
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import cross_val_score, GridSearchCV

# Scorer
f1_scorer = make_scorer(precision_score)
# SVC
svc = GridSearchCV( SVC(), param_grid={'C': [1, 10, 1e2, 1e3], 'kernel': ['linear', 'poly', 'rbf'],
                                       'degree': [2,3]}, 
                   scoring=f1_scorer)
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)
print('SVC Precision Score: {:.4f}'.format(precision_score(y_test, svc_pred)) )

# MLPClassifier
mlp = MLPClassifier(max_iter = 1000)
mlp.fit(x_train, y_train)
mlp_pred = mlp.predict(x_test)
print('MLP Precision Score: {:.4f}'.format(precision_score(y_test, mlp_pred)) )

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
print('RFC Precision Score: {:.4f}'.format(precision_score(y_test, rfc_pred)) )

# Voting classifier
clf = VotingClassifier(estimators=[('SVC', svc), ('MLP', mlp), ('RFC', rfc)], voting='hard').fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Voting Classifier Precision Score: {:.4f}'.format(precision_score(y_test, y_pred)) )