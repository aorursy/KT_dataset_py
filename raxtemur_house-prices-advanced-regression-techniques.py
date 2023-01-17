#!pip install --user --upgrade catboost

#!pip install --user --upgrade ipywidgets

#!pip install shap

#!pip install sklearn

#!pip install --upgrade numpy

!jupyter nbextension enable --py widgetsnbextension
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost

from catboost import *

from catboost import datasets

#from sklearn.metrics import mean_absolute_error

#from sklearn.ensemble import RandomForestRegressor

X = pd.read_csv('../input/train.csv', index_col='Id') 

#X.isnull().sum()

X.head()
X.describe()
y = X.SalePrice

X.drop(columns=['SalePrice'], axis=1, inplace=True)
y
s = (X.dtypes == 'object')

obj_cols = list(s[s].index)

X = X.drop(columns=obj_cols)



#cat_f_loc = list()

#for i in obj_cols: 

#    cat_f_loc.append(X.columns.get_loc(i))

#print(cat_f_loc)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='mean')



X_columns = X.columns

X = imputer.fit_transform(X)

X = pd.DataFrame(data=X[0:,0:], columns=X_columns)
#from sklearn.preprocessing import LabelEncoder

#enc = LabelEncoder()



#enc.fit(X['MSZoning'].values)



#for i in obj_cols:

#    enc.fit(X[i].values)

#    print(i)
import seaborn as sns

from scipy.stats import norm



sns.distplot(y, fit=norm);
y = np.log(y)

sns.distplot(y, fit=norm);
'''from catboost.utils import create_cd

import os

feature_names = dict()



for column, name in enumerate(X):

    if column == 0:

        continue

    feature_names[column - 1] = name



create_cd(

    label=0, 

    cat_features=cat_f_loc,

    feature_names=feature_names,

    output_path=os.path.join('./train.cd')

)'''
pool1 = Pool(data=X, label=y)
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size=0.8, random_state=1234)
from catboost import CatBoostRegressor

tunned_model = CatBoostRegressor(

    loss_function='RMSE',

    random_seed=63,

    iterations=3000,

    learning_rate=0.03,

    l2_leaf_reg=3,

    bagging_temperature=1,

    random_strength=1,

    one_hot_max_size=5,

    early_stopping_rounds=200

)

tunned_model.fit(

    X_train, y_train,

    cat_features=[],

    verbose=False,

    eval_set=(X_validation, y_validation),

    plot=True

)





'''params = {}

params['loss_function'] = 'Logloss'

params['iterations'] = 80

params['custom_loss'] = 'AUC'

params['random_seed'] = 63

params['learning_rate'] = 0.5



cv_data = cv(

    params = params,

    pool = 

    fold_count=5,

    shuffle=True,

    partition_random_seed=0,

    plot=True,

    stratified=False,

    verbose=False

)'''
X_test = pd.read_csv('../input/test.csv', index_col='Id') 
X_test.head()
X_test = X_test[X.columns]
predicted_y = tunned_model.predict(X_test)

predicted_y = np.exp(predicted_y)

predicted_y
predictions = pd.DataFrame({'Id': X_test.index, 'SalePrice' : predicted_y})

predictions.head()
predictions.to_csv('predictions.csv', index = False)