import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score

from sklearn.feature_selection import RFECV

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import minmax_scale,LabelEncoder

from sklearn.neighbors import KNeighborsRegressor

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

df.head(3)
def first_look(df):

    print('dataset shape: ',df.shape)

    print('dataset columns\n')

    print('-'*15)

    print(df.columns)

    print('data-type of each column: ')

    print('-'*15)

    print(df.dtypes)

    print('missing columns is : \n')

    c=df.isnull().sum()

    print(c[c>0])

first_look(df)
df=df.drop('Unnamed: 0',axis=1)

for i in df.columns:

    changer=i.replace('.',' ').strip().replace(' ','_').lower()

    df.rename(columns={i:changer},inplace=True)
stats=df.describe()

stats
plt.figure(figsize=(10,10))

plt.title('Correlation')

ax=sns.heatmap(df.corr(),

               linewidth=2.6,

               annot=True,

               center=1)
df['genre'].value_counts().plot.bar()

plt.title('Genre Distribution')

plt.ylabel('frequency')

plt.xlabel('Genre')

plt.show
fig=plt.figure(figsize=(20,25))

for i,j in zip(df.drop(['genre','track_name','artist_name'],axis=1).columns,range(10)):

    ax=fig.add_subplot(5,2,j+1)

    sns.distplot(df[i],ax=ax,axlabel=False)

    plt.axvline(df[i].mean(),label='mean',color='blue')

    plt.axvline(df[i].median(),label='median',color='green')

    plt.axvline(df[i].std(),label='std',color='red')

    plt.title('{} distribtion'.format(i))

    plt.legend()

plt.show()
fig=plt.figure(figsize=(20,25))

for i,j in zip(df.drop(['genre','track_name','artist_name'],axis=1).columns,range(10)):

    ax=fig.add_subplot(5,2,j+1)

    sns.boxplot(i,data=df,ax=ax,color='green')

plt.show()
def show_outliers(df,col): 

    outliers={}

    for j,k in enumerate(df[col].tolist()):

        iqr=stats.loc['75%',col]-stats.loc['25%',col]

        upper_bound=stats.loc['75%',col]+iqr*1.5

        lower_bound=stats.loc['25%',col]-iqr*1.5

        if k>upper_bound :

            outliers[k]=['upper',df.loc[j,'track_name'],df.loc[j,'artist_name']]

        elif k<lower_bound:

            outliers[k]=['lower',df.loc[j,'track_name'],df.loc[j,'artist_name']]

    print(outliers)

for i in df.drop(['genre','track_name','artist_name'],axis=1).columns:

    print(i)

    print('-'*10)

    show_outliers(df,i)
def drop_useless_cols(df,cols):

    df=df.drop(cols,axis=1)

    return df

df_1=drop_useless_cols(df,'track_name')
all_x=df_1.drop('popularity',axis=1)

all_y=df_1['popularity']
def handling_categories(df,column_names):

    for i in column_names:

        le=LabelEncoder()

        df[i]=le.fit_transform(df[i])

    return df

all_x=handling_categories(all_x,['artist_name','genre'])
def handling_numerical(df):

    for i in df.columns:

        df[i]=minmax_scale(df[i])

    return df

all_x=handling_numerical(all_x)
def select_features(x,y):

    xgb=RandomForestRegressor(random_state=0)

    selector=RFECV(xgb,cv=10)

    selector.fit(x,y)

    best_columns=list(x.columns[selector.support_])

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

        'name':'KNeighborsRegressor',

        'estimator': KNeighborsRegressor(),

        'hyperparameters':{

            "n_neighbors": range(1,20,2),

            "weights": ["distance", "uniform"],

            "algorithm": ["ball_tree", "kd_tree", "brute"],

            "p": [1,2]

        }

    },

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