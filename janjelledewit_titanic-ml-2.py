import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Below are three exploratory functions.



def col_types(df, target, limit=10): 

    '''

    This function distinguishes three different types of data:

        - Numerical data

        - Categorical data with low cardinality (< lim)

        - Categorical data with hight cardinality (> lim)

    Cardinality is the number values the categorical data takes.

    '''

    if target in df.columns:

        df.drop(target, axis=1, inplace=True)

    #

    num_cols = [p for p in df.columns 

                if df[p].dtype in ['int64','float64']]

    cat_cols_low  = [p for p in df.columns if (df[p].dtype == 'object' 

                                           and df[p].nunique() < limit )] 

    cat_cols_high = [p for p in df.columns if (df[p].dtype == 'object' 

                                           and df[p].nunique() > limit )]

    return num_cols, cat_cols_low, cat_cols_high



def explorer(df, target, limit=10):

    '''

    This function explores and prints all information necessary about a 

    dataset to start develop an effective pipeline.

    '''

    print('This is an exploration:  \n')

    print('Our dataset has:')

    print('{} rows and {} columns'.format(df.shape[0], df.shape[1]), '\n')

    print('Missing values:')

    a = df.isnull().sum()

    print( [[a.index[i], a[i]] for i in range( len(a) ) if a[i] > 0], '\n')

    print('Column types:')

    num_cols, cat_cols_low, cat_cols_high = col_types(df, target)

    print('Numerical columns: ', num_cols)

    print('Low cardinality categorical columns: ', cat_cols_low)

    print('High cardinality categorical columns: ', cat_cols_high)

    print('\n')

    pass



# We import the train_data and test_data

train_data = pd.read_csv('../input/titanic/train.csv',index_col=0)

test_data = pd.read_csv('../input/titanic/test.csv',index_col=0)





#And we explore them:

target = 'Survived'

explorer(train_data, target) 

explorer(test_data, target)



  
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor



#First we need to discern three types of columns

num_cols, cat_cols_low, cat_cols_high = col_types(train_data, 'Survived')



#An appropriate strategy for the 'Age' still needs to be found.

num_cols.remove('Age') 





numerical_transformer = SimpleImputer(strategy='mean') 



categorical_transformer = Pipeline([ 

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder())   ])



data_preprocessor = ColumnTransformer( transformers=[

    ('ignore', 'drop', cat_cols_high) ,     

    ('cat', categorical_transformer, cat_cols_low) , 

    ('num', numerical_transformer, num_cols)  ])





model = XGBRegressor(n_estimators=2500, learning_rate=0.01)



clf = Pipeline(steps=[ ('prep', data_preprocessor),

                      ('learn', model)    ])
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score





train_data = pd.read_csv('../input/titanic/train.csv',index_col=0 )



y = train_data.Survived

X = train_data.drop( ['Survived'] , axis=1 ) 



X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=0)



clf.fit(X_train, y_train)

preds = np.around( clf.predict(X_val) , decimals=0).astype(int)

res = accuracy_score(y_val, preds, normalize=False)



print('Of the {} entries, {} or {}% were predicted correctly'.format( 

               len(y_val), res, round(  res/len(y_val) *100, 1)))

    

test_data = pd.read_csv("../input/titanic/test.csv",index_col=0)

clf.fit(X, y)

preds = np.around( clf.predict(test_data) , decimals=0).astype(int)



output = pd.DataFrame({ 'PassengerId' : test_data.index, 

                       'Survived': preds 

                         })

output.to_csv('Test_predictions.csv', index=False)