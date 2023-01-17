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



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/winequality-red.csv')

df.head()
df.describe()
df.columns
df.shape
X=df.drop('quality',axis=1).values

y=df.quality.values
X=X.reshape(-1,11)

y=y.reshape(-1,)
# Import scale

from sklearn.preprocessing import scale



# Scale the features: X_scaled

X_scaled = scale(X)



# Print the mean and standard deviation of the unscaled features

print("Mean of Unscaled Features: {}".format(np.mean(X))) 

print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))



# Print the mean and standard deviation of the scaled features

print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 

print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))
# Import the necessary modules

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split



# Setup the pipeline steps: steps

steps = [('scaler', StandardScaler()),

        ('knn', KNeighborsClassifier())]

        

# Create the pipeline: pipeline

pipeline = Pipeline(steps)



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)



# Fit the pipeline to the training set: knn_scaled

knn_scaled = pipeline.fit(X_train,y_train)



# Instantiate and fit a k-NN classifier to the unscaled data

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)



# Compute and print metrics

print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test,y_test)))

print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test,y_test)))

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

# Setup the pipeline

steps = [('scaler', StandardScaler()),

         ('SVM', SVC())]



pipeline = Pipeline(steps)



# Specify the hyperparameter space

parameters = {'SVM__C':[1, 10, 100],

              'SVM__gamma':[0.1, 0.01]}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=21)



# Instantiate the GridSearchCV object: cv

cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)



# Fit to the training set

cv.fit(X_train,y_train)



# Predict the labels of the test set: y_pred

y_pred = cv.predict(X_test)



# Compute and print metrics

print("Accuracy: {}".format(cv.score(X_test, y_test)))

print(classification_report(y_test, y_pred))

print("Tuned Model Parameters: {}".format(cv.best_params_))

from sklearn.preprocessing import Imputer

from sklearn.linear_model import ElasticNet

# Setup the pipeline steps: steps

steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),

         ('scaler', StandardScaler()),

         ('elasticnet', ElasticNet())]



# Create the pipeline: pipeline 

pipeline = Pipeline(steps)



# Specify the hyperparameter space

parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}



# Create train and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=42)



# Create the GridSearchCV object: gm_cv

gm_cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)



# Fit to the training set

gm_cv.fit(X_train,y_train)



# Compute and print the metrics

r2 = gm_cv.score(X_test, y_test)

print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))

print("Tuned ElasticNet R squared: {}".format(r2))
