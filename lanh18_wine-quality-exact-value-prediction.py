# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import the data into table:

df=pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

#See the correlation betweeen feautures to get some insight about the data

plt.figure(figsize=(18,9))

sns.heatmap(df.corr(),annot=True)
set(df['quality'])
#Split the data into Train_Test set:

X=df[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]

y=df['quality']

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=18)
#Using linear model as a default metrics for evaluation of regression:

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train,y_train)

y_pred=reg.predict(X_test)

lreg_mse=mean_squared_error(y_test,np.around(y_pred))

lreg_accuracy=accuracy_score(y_test,np.around(y_pred))

print('Mean squared error: %.2f'%lreg_mse)

print('Accuracy Score: %.2f'%lreg_accuracy)
from sklearn.linear_model import LogisticRegression

log_reg=LogisticRegression()

log_reg.fit(X_train, y_train)

y_pred=log_reg.predict(X_test)

log_reg_mse=mean_squared_error(y_test,y_pred)

log_reg_accuracy=accuracy_score(y_test,y_pred)

print('Mean squared error: %.2f'%log_reg_mse)

print('Accuracy Score: %.2f'%log_reg_accuracy)
#Using grid search for better parameter tunning:

from sklearn.svm import SVC

C= [0.001, 0.01, 0.1, 1, 10]

gamma = [0.001, 0.01, 0.1, 1]

param_grid = {'C': C, 'gamma' : gamma}

svc_grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid,cv=7)

svc_grid_search.fit(X_train,y_train)

y_pred=svc_grid_search.predict(X_test)

svc_mse=mean_squared_error(y_test,np.around(y_pred))

svc_accuracy=accuracy_score(y_test,np.around(y_pred))

print('Mean squared error: %.2f'%svc_mse)

print('Accuracy Score: %.2f'%svc_accuracy)
#Since regression model not quite predict the true value, I decide to use multiclass classification.

from sklearn.ensemble import GradientBoostingClassifier

parameters = {

    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],

    "max_depth":[5,6,7],

    "max_features":["log2","sqrt"]

    }

grad_boosting_model=GridSearchCV(GradientBoostingClassifier(), parameters,cv=7)

grad_boosting_model.fit(X_train,y_train)

y_pred=grad_boosting_model.predict(X_test)

grad_boosting_accuracy=grad_boosting_model.score(X_test,y_test)

print('Accuracy Score: %.2f'%grad_boosting_accuracy)
#Another classifier:

from sklearn.ensemble import RandomForestClassifier

parameters = {

    "n_estimators":[100,200,300,400,500]}

forest_model=GridSearchCV(RandomForestClassifier(),parameters,cv=7)

forest_model.fit(X_train,y_train)

y_pred=forest_model.predict(X_test)

forest_accuracy=forest_model.score(X_test,y_test)

print('Accuracy Score: %.2f'%forest_accuracy)