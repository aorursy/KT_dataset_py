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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

### reading up the data



X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv')

## our target is to predict SalePrice of the houses

## so let's first preprocess our data



##  remove rows with missing target value 



X_full.dropna(axis=0,subset=['SalePrice'],inplace=True)

y = X_full.SalePrice # storing our target column in y



## now drop our target from X_full



X_full.drop(['SalePrice'],axis=1,inplace=True)



## split th data



X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)



categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.info()
X_train.describe()
## all the cols with missing values in X_train, which we need to handle

print([col for col in X_train.columns if X_train[col].isnull().any()])

print(categorical_cols)

print(numerical_cols)

### we will now preprocess our data using pipline

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import ElasticNet





scaler = StandardScaler()



numerical_transformer = SimpleImputer(strategy='constant')



categorical_transformer = Pipeline([

    ('impute',SimpleImputer(strategy='most_frequent')),

    ('onehot',OneHotEncoder(handle_unknown='ignore',sparse=False))

])





preprocessor = ColumnTransformer(transformers=[

    ('num',numerical_transformer,numerical_cols),

    ('cat',categorical_transformer,categorical_cols)

])





model_rf = RandomForestRegressor(n_estimators=100,random_state=0)

model_en = ElasticNet(random_state=0)

model_gb = XGBRegressor(n_estimators=200,learning_rate=0.05,random_state=0)





my_pipeline = Pipeline(steps=[

    ('pre',preprocessor),

    ('sc',scaler),

    ('gb',model_gb)

])



parameters = {'gb__n_estimators':[i for i in range(100,1000,100)]}











cv = GridSearchCV(my_pipeline,parameters,cv=10)
cv.fit(X_train,y_train)
preds = cv.predict(X_valid)
mae = mean_absolute_error(y_valid,preds)

mae
cv.best_params_
## for final submission we need to make sure that our test data is also preprocessed





test_pipeline = Pipeline([

    ('preprocessor',preprocessor)

    #,('model',model)

])





test_pipeline.fit(X_test)
preds_test = cv.predict(X_test)

output = pd.DataFrame({'Id': X_test.Id,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)