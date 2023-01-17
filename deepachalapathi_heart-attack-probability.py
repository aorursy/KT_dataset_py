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
dataset = pd.read_csv('../input/health-care-data-set-on-heart-attack-possibility/heart.csv')

dataset.info()

dataset.head()
dataset.isnull().values.any()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split as tst
X_train, X_test, y_train, y_test = tst(X, y, test_size=0.20, random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) #cv is the number of of folds
print("Acc.:{:.3f} %".format(accuracies.mean()*100))
print("Std. Dev.:{:.2f} %".format(accuracies.std()*100))
#allows to find the find the best version of a model by finding the optimal parameters
from sklearn.model_selection import GridSearchCV
#parameters is a list of hyper-parameters. two dictionries for rbf and linear 
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel':['linear']},
             {'C':[0.25, 0.5, 0.75, 1], 'kernel':['rbf', 'poly', 'sigmoid'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier, 
                           param_grid = parameters,
                           scoring = 'accuracy', #ig it's r-squared for regression models
                           cv = 10,
                           n_jobs = -1) #n_jobs = to specify all the processors on the system to run, optional
grid_search.fit(X_train,y_train)
best_acc = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_acc*100))
print("Best Parameters:", best_parameters)
#ann
import tensorflow as tf

def create_network():
    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=32, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return ann
from keras.wrappers.scikit_learn import KerasClassifier
ann = KerasClassifier(build_fn=create_network, 
                                 epochs=10, 
                                 batch_size=10, 
                                 verbose=1)
from sklearn.model_selection import cross_val_score
cross_val_score(ann,X_train,y_train, cv=10)
from sklearn.model_selection import cross_val_score
cross_val_score(ann,X_test,y_test, cv=10)