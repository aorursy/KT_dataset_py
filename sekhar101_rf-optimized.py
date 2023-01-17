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
#from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/home-data-for-ml-course/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/home-data-for-ml-course/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)





# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_full.columns if

                    X_full[cname].nunique() < 10 and 

                    X_full[cname].dtype == "object"]



#Add the 3 categorical variables with high cardinality for added features

hi_cardinality_cols = [cname for cname in X_full.columns if

                    X_full[cname].nunique() >= 10 and 

                    X_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_full.columns if 

                X_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols + hi_cardinality_cols

X_train = X_full[my_cols].copy()

# X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
X_train.head(5)
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols),        

    ])



# Define model

model = RandomForestRegressor(n_estimators=425, random_state=10, max_features=40)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])







# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(clf, X_train, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())



scores

#16896.400310958903, 500, 30

#16967.52878162772, 425, 25

#16879.062887993554, 425, max_features 40 

#16656.24277679291, same as above with CV 5
#This code will identify the best number for n_estimators

#--------------------------------------------------------

# def get_score(n_estimators):

#     my_pipeline = Pipeline(steps=[

#     ('preprocessor', preprocessor),

#     ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))])

    

#     scores = -1 * cross_val_score(my_pipeline, X_train, y,

#                               cv = 3,

#                               scoring='neg_mean_absolute_error')



#     return scores.mean()

#     # Replace this body with your own code

# pass



# results = {}

# for i in range(4,18):

#     results[25*i] = get_score(25*i)

    

    
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()



#Shows best n_estimators = 425
# Preprocessing of test data, fit model



model = clf.fit(X_train, y)



preds_test = model.predict(X_test) 
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
#For feature engineering on hi_cardinality columns

hi_cardinality_cols = [cname for cname in X_full.columns if

                    X_full[cname].nunique() >= 10 and 

                    X_full[cname].dtype == "object"]



hi_cardinality_cols
X_train.shape
X_full['Neighborhood'].value_counts().plot(kind='bar')
def valueToFlag(value):

    if   value == 'NAmes': return 1

    elif value == 'CollgCr': return 2

    elif value == 'OldTown': return 3

    elif value == 'Edwards': return 4

    elif value == 'Somerst': return 5

    elif value == 'Gilbert': return 6

    elif value == 'NridgHt': return 7

    elif value == 'Sawyer': return 8

    elif value == 'BrkSide': return 9

    else: return 0

  

X_full['Neighborhood'] = X_full['Neighborhood'].apply(valueToFlag)

X_full["Neighborhood"].value_counts()
def valueToFlag1(value):

    if   value == 'VinylSd': return 1

    elif value == 'HdBoard': return 2

    elif value == 'MetalSd': return 3

    elif value == 'Wd Sdng': return 4

    elif value == 'Plywood': return 5

    elif value == 'CemntBd': return 6

    elif value == 'BrkFace': return 7

    else: return 0

  

X_full['Exterior1st'] = X_full['Exterior1st'].apply(valueToFlag1)

X_full["Exterior1st"].value_counts()
def valueToFlag2(value):

    if   value == 'VinylSd': return 1

    elif value == 'HdBoard': return 2

    elif value == 'MetalSd': return 3

    elif value == 'Wd Sdng': return 4

    elif value == 'Plywood': return 5

    elif value == 'CemntBd': return 6

    elif value == 'BrkFace': return 7

    elif value == 'Stucco': return 8

    elif value == 'Wd Shng': return 9

    else: return 0

  

X_full['Exterior2nd'] = X_full['Exterior2nd'].apply(valueToFlag2)

X_full["Exterior2nd"].value_counts()
X_full["Exterior2nd"].value_counts()