import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import minmax_scale

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')

df.head(3)
def first_look(df):

    print('dataset shape: ',df.shape)

    print('dataset columns\n')

    print('-'*15)

    print(df.columns)

    print('data-type of each column: ')

    print('-'*15)

    print(df.dtypes)

    print('column distributions %:')

    for i in df.columns:

        print(i)

        print('-'*10)

        print(df[i].value_counts(normalize=True)*100)

    print('missing columns is : \n')

    c=df.isnull().sum()

    print(c[c>0])
first_look(df)
all_x=df

all_y=df['Chance of Admit ']

all_x.drop(['Chance of Admit ','Serial No.'],axis=1,inplace=True)

print('it is Done!')
def change_col_names(df):

    for i in df.columns:

        changer=i.strip().lower().replace(' ','_')

        df.rename(columns={i:changer},inplace=True)

change_col_names(all_x)
lst=['gre_score','toefl_score','sop','lor','cgpa']

def rescale(data,cols_lst):

    for i in cols_lst:

        data[i]=(data[i]-data[i].min())/(data[i].max()-data[i].min())

    return data

rescale(all_x,lst)
def select_model(x,y):

    models=[{

        'name':'LinearRegression',

        'estimator':LinearRegression(),

        'hyperparameters':{

        }

    },

    {

        'name':'RandomForestRegressor',

        'estimator':RandomForestRegressor(),

        'hyperparameters':{

            "n_estimators": [4, 6, 9],

            "max_depth": [2, 5, 10],

            "max_features": ["log2", "sqrt"],

            "min_samples_leaf": [1, 5, 8],

            "min_samples_split": [2, 3, 5]

    }},

    {

        'name':'ExtermeGradientBoost',

        'estimator':XGBRegressor(),

        'hyperparameters':{

            'n_estimators':[800,1000],

            'learning_rate':[0.05],

            'n_jobs':[5]

        }

    }    

        

    ]

    for i in models:

        print(i['name'])

        grid=GridSearchCV(i['estimator'],param_grid=i['hyperparameters'],cv=10,scoring='neg_mean_absolute_error')

        grid.fit(x,y)

        i["best_params"] = grid.best_params_

        i["best_score"] = grid.best_score_

        i["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(i["best_score"]))

        print("Best Parameters: {}\n".format(i["best_params"]))



    return models



result = select_model(all_x,all_y)