import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV

from sklearn.feature_selection import RFECV

from sklearn.preprocessing import minmax_scale, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

import os

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_df=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
def explore_data(df):

    print('Dataset shape:\n')

    print(df.shape)

    print('columns in my data:\n')

    print(df.columns)

    print('Data-type of each columns:\n')

    print(df.dtypes)

    print('frequency distribution of each column:\n')

    for i in df.columns:

        print(i)

        print('-'*20)

        print(df[i].value_counts(normalize=True)*100)

        print('-'*40)

    print('missing values in column:\n')

    null_cols=(df.isnull().sum()*100)/len(df)

    print(null_cols[null_cols!=0])
explore_data(train)
explore_data(test)
def drop_some_columns(df):

    null_cols=(df.isnull().sum()*100)/len(df)

    cols_to_drop=null_cols[null_cols>40]

    df=df.drop(cols_to_drop.index,axis=1)

    return df

train=drop_some_columns(train)

test=drop_some_columns(test)
def drop_useless_cols(df,cols):

    df=df.drop(cols,axis=1)

    return df

train=drop_useless_cols(train,'Id')

test=drop_useless_cols(test,'Id')
train.dropna(axis=0, subset=['SalePrice'], inplace=True)
all_x=train

all_y=all_x['SalePrice']

all_x.drop(['SalePrice'], axis=1, inplace=True)

categorical_cols=[i for i in all_x.columns if all_x[i].dtype=='object' and all_x[i].nunique()<10]

numerical_cols=[i for i in all_x.columns if all_x[i].dtype in ['int','float']]

object_cols = [col for col in categorical_cols if all_x[col].dtype == "object"]

good_label_cols = [col for col in object_cols if 

                   set(all_x[col]) == set(test[col])]

bad_label_cols = list(set(object_cols)-set(good_label_cols))

data=good_label_cols+numerical_cols

all_x=all_x[data]

test=test[data]
def handling_categories(df,column_names):

    for i in column_names:

        c=df[i].value_counts()

        df[i]=df[i].fillna(c.index.max())

        dummies = pd.get_dummies(df[i],prefix=i)

        df = pd.concat([df,dummies],axis=1)

        df=df.drop(i,axis=1)

    return df

all_x=handling_categories(all_x,good_label_cols)

test=handling_categories(test,good_label_cols)
def handling_numerical(df,column_names):

    for i in column_names:

        df[i]=df[i].fillna(df[i].mean())

    return df

all_x=handling_numerical(all_x,numerical_cols)

test=handling_numerical(test,numerical_cols)
sns.distplot(all_y)

plt.title('Distribution of Sale Prices')

plt.axvline(all_y.mean(),label='mean',color='red')

plt.axvline(all_y.median(),label='median',color='green')

plt.axvline(all_y.std(),label='std',color='blue')

plt.legend()

plt.show()
def select_features(x,y):

    xgb=XGBRegressor(random_state=0)

    selector=RFECV(xgb,cv=10)

    selector.fit(x,y)

    best_columns = list(x.columns[selector.support_])

    print("Best Columns \n"+"-"*12+"\n{}\n".format(best_columns))

    

    return best_columns

cols=select_features(all_x,all_y)   
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

        grid=GridSearchCV(i['estimator'],

                          param_grid=i['hyperparameters'],

                          cv=10,scoring='neg_mean_absolute_error')

        grid.fit(x,y)

        i["best_params"] = grid.best_params_

        i["best_score"] = grid.best_score_

        i["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(i["best_score"]))

        print("Best Parameters: {}\n".format(i["best_params"]))



    return models



result = select_model(all_x[cols],all_y)
xgb=XGBRegressor(n_estimators=1000,learning_rate=0.05,n_jobs=5)

xgb.fit(all_x[cols],all_y)

pred=xgb.predict(test[cols])

pred[:15]
submission=pd.DataFrame({'Id':test_df['Id'],'SalePrice':pred})

submission.to_csv('submission.csv',index=False)