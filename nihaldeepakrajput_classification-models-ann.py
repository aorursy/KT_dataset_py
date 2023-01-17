#import all packages
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head(10)
df.dtypes
df.isnull().sum()
df['oldpeak'] = df['oldpeak'].astype(int)
df.describe().T
X = df.drop('target', axis = 1).values
y = df['target'].values
#split into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#scale the entire dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
#Feature Extraction - PCA
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
#LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
classifier = LogisticRegression(solver = 'lbfgs')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
#KNN Classification
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
#SVC
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)
#Model selection using K fold
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
#max accuracy 
cvs.max()
#standard deviation
cvs.std()
#model selection using Grid Search
classifier = SVC()
from sklearn.model_selection import GridSearchCV
param = [{'C':[1,10,100,1000], 'kernel': ['linear']},
         {'C':[1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.1,0.01,0.001,0.0001]},
         {'C':[1,10,100,1000], 'kernel': ['sigmoid'], 'gamma': [0.1,0.01,0.001,0.0001]}]
gvs = GridSearchCV(estimator = classifier, cv = 10, n_jobs = -1, 
                   param_grid = param, scoring = 'accuracy')
gvs = gvs.fit(X_train, y_train)
gvs.best_score_
gvs.best_params_
#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 7, activation = 'relu', kernel_initializer = 'uniform', input_dim = 4))
classifier.add(Dense(units = 7, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(units = 7, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 150)
y_pred = classifier.predict(X_test)
y_pred = y_pred>0.7
accuracy_score(y_test, y_pred)