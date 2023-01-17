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
import pandas as pd #importing all the necessary libraries

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() #loading our data
cancer.keys()
df = pd.DataFrame(cancer['data'],columns = cancer['feature_names']) #putting our data in a Dataframe
df.head() #checking the head of the data
df.describe()
df.isnull().sum() #checking for any sort of null value in our data
sns.heatmap(df.isnull()) #looking for null values with help of heat map
from sklearn.model_selection import train_test_split #to split our data into training and testing set
x = df

y = cancer['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 101) #splitting our data
from sklearn.svm import SVC #importing the svm model
svc  =SVC()
svc.fit(x_train,y_train) #fitting the data to our model
pred = svc.predict(x_test) #predicting the result
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))

print('\n')

print(confusion_matrix(y_test,pred)) #we print our results and its quite decent but it can be improved by using GridSearch which would help us find better hyperparameters for our problem
from sklearn.model_selection import GridSearchCV #importing Gridsearch
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose = 3)
grid.fit(x_train,y_train)
grid.best_estimator_
grid.best_params_
grid_pred = grid.predict(x_test)
print(classification_report(y_test,grid_pred))

print('\n')

print(confusion_matrix(y_test,grid_pred))