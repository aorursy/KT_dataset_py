import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("Num Cols: ", len(df.columns))

print ("Num Rows: ",len(df.index))
df.head()
df.info(verbose=True, null_counts=True)
df.describe()
ax = sns.heatmap(df.isnull(),yticklabels=False,cbar=False)

ax.set(xlabel='columns', ylabel='rows (white if null)')

plt.show()
fig=plt.figure(figsize=(25, 5))

unique = df.select_dtypes(include=['object','category']).nunique().sort_values()

plt.bar(unique.index, unique)

plt.xticks(rotation=90)

plt.show()

print("min: ", unique.min())

print("max: ", unique.max())
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# Define feature column categories by column type

categorical_cols = df.select_dtypes(include=['object','category']).columns.to_list()

numeric_cols = df.select_dtypes(include='number').columns.to_list()

# Remove the target column (SalePrice) from our feature list

numeric_cols.remove('SalePrice')



print ("Categorical columns: ", categorical_cols)

print ("Numeric columns: ", numeric_cols)
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')
# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])
# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('numeric', numerical_transformer, numeric_cols),

        ('categorical', categorical_transformer, categorical_cols)

    ])
from sklearn.model_selection import train_test_split



# Grab target as y, remove target from X

train_test = df.copy()

y = train_test.SalePrice

X = train_test.drop(columns=['SalePrice'])



# Split into train, test

train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, random_state = 17)
train_X.head()
train_y.head()
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score



# Time to tune params!

def display_validation(pipeline):

    # Preprocessing of training data, fit model 

    pipeline.fit(train_X,train_y)

    # Preprocessing of validation data, get predictions

    preds = pipeline.predict(val_X)



    # Evaluate the model

    score = mean_absolute_error(val_y, preds)

    print('MAE:', score)
from sklearn.ensemble import RandomForestRegressor

import random



for n in [50,100, 500]:

    model = RandomForestRegressor(n_estimators=n, random_state = 17)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])

    print("n_estimators: ", n)

    display_validation(pipeline)
# First, train again on that best n_estimators value

final_model = RandomForestRegressor(n_estimators=100, random_state = 17)

final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', final_model)

                             ])



# Preprocessing of validation data, get predictions

final_pipeline.fit(train_X,train_y)

test_data_labels = final_pipeline.predict(test)



# Create predictions to be submitted!

pd.DataFrame({'Id': test.Id, 'SalePrice': test_data_labels}).to_csv('RFC_100.csv', index =False)  

print("Done :D")