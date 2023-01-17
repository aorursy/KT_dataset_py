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

        

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





#SK-Learn        

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)



from pprint import pprint

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR,LinearSVR

from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import RFE, f_regression

from sklearn.model_selection import GridSearchCV



import time



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

from pathlib import Path

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
PATH=Path("/kaggle/input/graduate-admissions")

path_11=PATH/"Admission_Predict_Ver1.1.csv"
df=pd.read_csv(path_11,index_col='Serial No.')
df.head()
df.info()
X=df.copy()

y=df['Chance of Admit ']

X.drop('Chance of Admit ',axis=1,inplace=True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.16, random_state=1)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
mean = X_train.mean()

std = X_train.std()
X_train = (X_train-mean)/std

X_test = (X_test-mean)/std
print(std,mean)
X_train.head(3)
models = pd.DataFrame(columns=['model', 'score','Time to Train'])





options = [

           SVR(), 

           LinearSVR(),

           SGDRegressor(), 

           XGBRegressor()]   



model_names = [ 

               'Support Vector Machine', 

               'Linear SVC', 

               'SGD Regressor',

               'XGBoost']  



for (opt, name) in zip(options, model_names):

    start=time.time()

    model = opt

    print(model)

    model.fit(X_train, y_train)

    scores = model.score(X_test,y_test)

    end=time.time()

    row = pd.DataFrame([[name, scores.mean(), end-start]], columns=['model', 'score','Time to Train'])

    models = pd.concat([models, row], ignore_index=True)



models.sort_values(by='score', ascending=False)

model = SGDRegressor(random_state = 42)

model.fit(X_train, y_train)
SGD = RFE(model, n_features_to_select=1, verbose =3)

SGD.fit(X_train,y_train)



imp1 = pd.DataFrame({'feature':X_train.columns, 'rank1':SGD.ranking_})

imp1 = imp1.sort_values(by = 'rank1')

imp1
df.corr()['Chance of Admit ']
print('Parameters currently in use:\n')

pprint(SGD.get_params())
SGD_tmp=SGDRegressor(random_state=42)

param_grid={

    'loss': ['squared_loss','huber'],

    'penalty':['l1','l2'],

    'validation_fraction':[0.1,0.12,0.15],

    'learning_rate':['optimal','invscaling','adaptive']

}

pprint(param_grid)
SGD_grid=GridSearchCV(estimator=SGD_tmp,param_grid=param_grid,cv=5)

SGD_grid.fit(X_train,y_train)
SGD_grid.best_params_
SGD_tuned=SGDRegressor(loss='huber',penalty='l2',learning_rate='adaptive',validation_fraction=0.1)

SGD_tuned.fit(X_train,y_train)
print("Accuracy for Random Forest after Hyperparameter Tuning on test data: ",SGD_tuned.score(X_test,y_test))

print("Accuracy for Random Forest before Hyperparameter Tuning on test data: ",SGD.score(X_test,y_test))
import pickle

Pkl_Filename = "Pickle_SGD_Model.pkl"  



with open(Pkl_Filename, 'wb') as file:  

    pickle.dump(SGD_tuned, file)