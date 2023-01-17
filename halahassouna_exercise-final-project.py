import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

data = pd.read_csv('../input/titanichala/train.csv')



# Separate target from predictors

y = data.Survived

X = data.drop(['Survived'], axis=1)



# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()
data
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
# Check for a dataset with a CSV file

#step_1.check()
# # Fill in the line below: Specify the path of the CSV file to read

# my_filepath = "../input/titanichala/train.csv"



# # Check for a valid filepath to a CSV file in a dataset

# step_2.check()
# # Fill in the line below: Read the file into a variable my_data

# my_data = pd.read_csv(my_filepath)

# test_data = pd.read_csv("../input/titanichala/test.csv")



# # Check that a dataset has been uploaded into my_data

# step_3.check()
# Print the first five rows of the data

data.head()

from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(n_estimators=100, random_state=0)
# Create a plot

# Your code here

plt.figure(figsize= (16,6))

sns.barplot(data= data)

# Check that a figure appears below

step_4.check()
plt.figure(figsize=(16,6))

sns.barplot(x='Survived', y='Sex', data= data)
sns.scatterplot(x=data['Survived'], y=data['Age'])
sns.scatterplot(data=data, x='Survived', y='Sex')
sns.lmplot(data=data, x='Survived', y='Sex')
sns.swarmplot(data=data, x='Survived', y='Sex')
sns.distplot(a=data['Survived'], kde= False)
sns.kdeplot(data= data['Survived'], shade= True)