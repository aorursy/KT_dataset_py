# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from sklearn.linear_model import LinearRegression,Ridge, Lasso

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

# Separate target from predictors

data=pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

y = data.SalePrice

X = data.drop(['SalePrice'], axis=1)

#df = test.drop(columns = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'])

# Divide data into training and validation subsets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001,

                                                                random_state=42)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and 

                       X_train[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_traina = X_train[my_cols].copy()

X_testa = X_test[my_cols].copy()



#Similar to how a pipeline bundles together preprocessing and modeling steps,

#we use the ColumnTransformer class to bundle together different preprocessing steps. The code below:



#imputes missing values in numerical data, and

#imputes missing values and applies a one-hot encoding to categorical data.

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])





# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=500, random_state=3)

#Finally, we use the Pipeline class to define a pipeline that bundles the preprocessing and modeling steps. There are a few important things to notice:



#With the pipeline, we preprocess the training data and fit the model in a single line of code. 

#(In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps. 

#This becomes especially messy if we have to deal with both numerical and categorical variables!)

#With the pipeline, we supply the unprocessed features in X_valid to the predict() command, 

#and the pipeline automatically preprocesses the features before generating predictions. 

#(However, without a pipeline, we have to remember to preprocess the validation data before making predictions.)

from sklearn.metrics import mean_absolute_error

#TestId=df['Id']

#align data set shapes and get dummies

#total_features=pd.concat((data.drop(['Id','SalePrice'], axis=1), df.drop(['Id'], axis=1)))

#total_features=pd.get_dummies(total_features, drop_first=True)

#train_features=test[0:data.shape[0]]



#making sure the test set matches the train set

test_features=test

# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_test)

preds

# Evaluate the model

score = mean_absolute_error(y_test, preds)

print('MAE:', score)

import pandas as pd





preds=my_pipeline.predict(test_features)

output = pd.DataFrame({'Id': test.Id,

                     'SalePrice': preds})

output.to_csv('submission.csv', index=False)



# Any results you write to the current directory are saved as output.

%ls '../input'
import pandas as pd

test=pd.read_csv('../input/test.csv')


