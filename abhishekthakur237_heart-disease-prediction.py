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
dataset = pd.read_csv('/kaggle/input/heart-disease/heart.csv')

dataset.head()
dataset.columns
categorical = ['sex', 'cp', 'restecg', 'slope', 'thal']

do_not_touch = ['fbs', 'exang']

non_categorical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical)],remainder='passthrough')

X = ct.fit_transform(dataset[categorical+do_not_touch+non_categorical])

y = dataset['target'].values
X[0,:]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=1)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[:,-6:] = scaler.fit_transform(X_train[:,-6:])

X_test[:,-6:] = scaler.transform(X_test[:,-6:])
X_train[0,:]
from sklearn.svm import SVC

estimator = SVC()



parameters = [{'kernel':['rbf'],

               'C':[1,10,100,1000],

               'gamma':[1,0.1,0.001,0.0001],

            },

            {'kernel':['poly'],

               'C':[1,10,100,1000],

               'gamma':[1,0.1,0.001,0.0001],

             'degree':range(1,5)}

             ]



from sklearn.model_selection import GridSearchCV

 

grid_search = GridSearchCV(

    estimator=estimator,

    param_grid=parameters,

    scoring = 'accuracy',

    n_jobs = 10,

    cv = 10,

    verbose=True

)

grid_search.fit(X_train, y_train)

grid_search.best_estimator_
y_pred = grid_search.best_estimator_.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(y_test,y_pred))

accuracy_score(y_test,y_pred)