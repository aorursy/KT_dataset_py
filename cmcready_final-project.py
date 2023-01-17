# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # let us split our training data so we can validate it 

from sklearn.preprocessing import LabelEncoder # this is used to categorize the occupation feature

from sklearn.compose import ColumnTransformer # used for bundling preprocessing steps in pipelining 

from sklearn.pipeline import Pipeline # our pipeline

from sklearn.impute import SimpleImputer # used to fill in missing values

from xgboost import XGBRegressor # model we will be using for our project

from sklearn.metrics import mean_absolute_error # import MAE

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

        

# get the file path of the data 

train_path = '/kaggle/input/cat-in-the-dat/train.csv'

test_path = '/kaggle/input/cat-in-the-dat/test.csv'



# make the dataframes from the csv file path

train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)





# list of features that we will be looking at  



features = ['bin_0', 'bin_1', 'bin_2','bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month']



# get the variables we will be looking at 

X = train_data[features] 



# the prediction target for our training data. 

y = train_data.target



# split our training data into different data frames for the training set and the validation set.

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 0)







# select object type columns into a list 

categorical_cols = [column for column in train_X.columns if

                    train_X[column].nunique() < 10 and train_X[column].dtype == "object"]





# Select numerical columns into a list 

numerical_cols = [column for column in train_X.columns if 

                train_X[column].dtype == 'int64' or train_X[column].dtype == 'float64']







# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# create our preprocessor for our pipeline 

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])





model = XGBRegressor(n_estimators=500, learning_rate=0.1, n_jobs=4)



# Bundle preprocessing and modeling code in a pipeline

pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                       ('model', model)

                      ])



# # fit to training data  

pipeline.fit(train_X, train_y)





# Apply processing to the val_X dataframe so it's formatted correctly

val_X = preprocessor.transform(val_X)



# fit the pipeline to all of our data now 

pipeline.fit(train_X, train_y, model__early_stopping_rounds=20, model__eval_set=[(val_X, val_y)])



# make the features for our actual test data 

test_X = test_data[features] 



# get the predictions from our data

predictions = pipeline.predict(test_X)



output = []



for i in predictions:

    output.append(i)



data = {'id' :test_data.id, 'target' :output}



# make the output dataframe from our data 

output_df = pd.DataFrame(data)   

print(output_df)

    

# make the output a csv file

output_df.to_csv('Submission.csv',index=False)



# Any results you write to the current directory are saved as output.