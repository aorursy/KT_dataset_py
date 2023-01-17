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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import cufflinks as cf

cf.go_offline()

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

type(cancer)
cancer.keys()
print(cancer["DESCR"])

#Here we get overall description of the data

print(cancer["feature_names"]) #The names of the attributes are listed here
print(cancer["data"])#The data of the features are listed here

#The next step is to combine data with the feature names and make a pandas dataframe
features=pd.DataFrame(cancer["data"],columns=cancer["feature_names"])

features.head()
features.info() #Here we get an overall picture of our data
cancer["target"] #This is our target data that is listed 1 or 0 which represent malignant and benign tumors
features.describe()
plt.figure(figsize=(15,10))

sns.distplot(cancer["target"])
X=features

y=cancer["target"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.svm import SVC

svm_model=SVC()

svm_model.fit(X_train,y_train)
predictions=svm_model.predict(X_test)

predictions
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

#here we get the classification report to learn how accurate our model is
print(confusion_matrix(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid={"C":[1,10,100,100],"gamma":[1,0.1,0.01,0.001,0.0001]} 

#here we select values for grid search to try

grid=GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train) # we apply it to our training data to see the best C and gamma values

#grid.fit() will find the best combination of C and gamma values for our model
grid.best_params_
grid.best_estimator_
grid_predictions=grid.predict(X_test) 

#Now we predict with this readjustment
print(classification_report(y_test,grid_predictions))
print(confusion_matrix(y_test,grid_predictions))