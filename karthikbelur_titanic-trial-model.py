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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

#train_data.head()
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

#test_data.head()
train_data.dropna(axis=0, subset=['Survived'], inplace=True)

y = train_data.Survived

train_data.drop(['Survived'], axis=1, inplace=True)

#print(train_data.shape)

#print(y.shape)
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error





# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(train_data, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [col for col in X_train_full.columns if 

                   X_train_full[col].nunique() < 10 and

                   X_train_full[col].dtypes == "object"]



# Select numerical columns

numerical_cols = [col for col in X_train_full.columns if

                 X_train_full[col].dtypes in ["int64", "float64"]]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = test_data[my_cols].copy()



#print(X_train.shape)

#print(X_valid.shape)

#print(X_test.shape)
# Now apply simpleImputer for numerical columns

numerical_trans = SimpleImputer(strategy = 'constant')



# Now apply simpleimputer and onehotencoder to categorical column

categorical_trans = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'constant')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))

])



#Now bundle preprocessing on both Numerical and Categorical into one

preprocessor  = ColumnTransformer(transformers = [

    ('num',numerical_trans, numerical_cols ),

    ('cat', categorical_trans, categorical_cols)

])
from sklearn.ensemble import RandomForestClassifier



#Now Define the model

model = RandomForestClassifier(n_estimators = 100, max_depth = 8, random_state = 0)



#Now bundle preprocessing and model into one code into a pipeline

my_pipeline = Pipeline(steps = [

    ('preprocessing', preprocessor),

    ('model', model)

])



# Preprocessing of training data, fit model 

my_pipeline.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = my_pipeline.predict(X_valid)



# Evaluate the model

score = mean_absolute_error(y_valid, preds)

#print('MAE:', score)



# Preprocessing of test data, fit model

preds_test = my_pipeline.predict(X_test) # Your code here

#print(preds_test)



output = pd.DataFrame({'PassengerId' : X_test.PassengerId, 'Survived' : preds_test})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")