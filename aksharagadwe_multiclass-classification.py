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
data= pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv',usecols=[0,1,3,4,5,6,7,8,9])

target = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv',usecols=[2])
data.info()

target.info()
data.head()
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()



data[["chol","trestbps","thalach","oldpeak"]]=scaler.fit_transform(data[["chol","trestbps","thalach","oldpeak"]])

data.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(data,target.values,test_size=0.2,random_state=42)

print(X_train.shape, X_test.shape , y_train.shape , y_test.shape)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier()

params = {

    'n_estimators' : [35,45,55],

    'criterion': ['gini', 'entropy'],

    'class_weight': ['balanced', 'balanced_subsample']

    

    }

grid_kn = GridSearchCV(estimator = rf,

                        param_grid = params,

                        scoring = 'accuracy', 

                        cv = 5, 

                        verbose = 1,

                        n_jobs = -1)



grid_kn.fit(X_train, y_train)



print(grid_kn.best_estimator_)

print(grid_kn.score(X_test, y_test))
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

nb= GaussianNB()

nb.fit(X_train, y_train.ravel())

accuracy = nb.score(X_test, y_test) 

print(accuracy) 



y_pred = nb.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred) 

print(cm2)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

params = {

    'criterion': ['gini', 'entropy'],

    'splitter' : ['best','random'],

    'max_features' : ['auto','sqrt','log2', None]

    

    }

grid_dt = GridSearchCV(estimator = dt,

                        param_grid = params,

                        scoring = 'accuracy', 

                        cv = 5, 

                        verbose = 1,

                        n_jobs = -1)



grid_dt.fit(X_train, y_train)



print(grid_dt.best_estimator_)

print(grid_dt.score(X_test, y_test))

from sklearn.naive_bayes import BernoulliNB

bnb= BernoulliNB()



bnb.fit(X_train,y_train.ravel())

accuracy_bnb = bnb.score(X_test,y_test)

print(accuracy_bnb)





y_pred = bnb.predict(X_test)

cm3 = confusion_matrix(y_test, y_pred) 

print(cm3)
from sklearn.linear_model import LogisticRegressionCV



lrcv=LogisticRegressionCV()



params_lrcv ={

    'multi_class' : ['auto','ovr','multinomial'],

    'solver' :['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],

    'penalty' :['l1', 'l2', 'elasticnet']

}



grid_lcv= GridSearchCV(estimator= lrcv,

                       param_grid=params_lrcv,

                       scoring = 'accuracy', 

                        cv = 5, 

                        verbose = 1,

                        n_jobs = -1

                      )

grid_lcv.fit(X_train, y_train)



print(grid_lcv.best_estimator_)

print(grid_lcv.score(X_test, y_test))



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)
grid_lcv.fit(X_train_pca, y_train)



print(grid_lcv.best_estimator_)

print(grid_lcv.score(X_test_pca, y_test))
