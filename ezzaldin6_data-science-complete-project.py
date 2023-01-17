import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd 

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, MinMaxScaler, FunctionTransformer

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import LinearRegression, Lasso

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.feature_selection import RFECV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

import category_encoders as ce

import re

import string as s

import os

plt.style.use('ggplot')

sns.set(palette='RdBu', context='notebook', style='darkgrid')

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
cars=pd.read_csv('/kaggle/input/automobile-dataset/Automobile_data.csv')

cars.head()
cars.info()
cars.describe()
cars.describe(exclude=['int', 'float'])
to_num=['normalized-losses', 'bore','stroke','horsepower','peak-rpm','price']

cars[to_num]=cars[to_num].replace('?','0')

cars[['horsepower','peak-rpm','price', 'normalized-losses']]=cars[['horsepower','peak-rpm','price','normalized-losses']].astype('float')

cars[['bore','stroke']]=cars[['bore','stroke']].astype('float')

cars[to_num]=cars[to_num].replace(0, np.nan)

cars.dropna(subset=['price'], inplace=True)

cars['num-of-doors']=cars['num-of-doors'].replace('?', 'four')
fig=plt.figure(figsize=(20, 8))

cat_cols=['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']

for i,j in zip(cars[cat_cols].columns, range(10)):

    ax=fig.add_subplot(2,5,j+1)

    c2=cars[i].value_counts()

    ax.bar(c2.index, c2)

    ax.set_title('bar plot of {}'.format(i))

    ax.set_xlabel(i)

    ax.set_xticklabels(c2.index,rotation=90)

plt.subplots_adjust(hspace=0.7)

plt.show()
fig=plt.figure(figsize=(25, 8))

cat_cols=['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']

for i,j in zip(cars[cat_cols].columns, range(10)):

    ax=fig.add_subplot(2,5,j+1)

    c1=pd.pivot_table(index=i, values='price', data=cars, aggfunc='mean')

    ax.plot(c1.index, c1['price'])

    ax.scatter(c1.index, c1['price'])

    ax.set_title('mean price for each {}'.format(i))

    ax.set_xlabel(i)

    ax.set_xticklabels(c1.index,rotation=90)

plt.subplots_adjust(hspace=0.8)

plt.show()
sns.pairplot(cars, diag_kind='hist')

plt.show()
corr_df=cars.corr()

mask=np.triu(np.ones_like(corr_df, dtype=bool))

plt.figure(figsize=(10,5))

sns.heatmap(corr_df,

            annot=True,

            mask=mask,

            fmt='.2f',

            linewidth=1)

plt.show()
#detecting outliers

for i in cars.columns:

    if cars[i].dtype!='object':

        print(i)

        mu=cars[i].mean()

        st=cars[i].std()

        cut_off=st * 3

        lower, upper=mu-cut_off, mu+cut_off

        print('number of outliers',len(cars[(cars[i]>upper)&(cars[i]<lower)]))

        print('\n')

        
cols_to_drop=['city-mpg', 'engine-location']

cars.drop(cols_to_drop, axis=1, inplace=True)

cars_1=cars.copy()
#x=x.drop('symboling', axis=1)
all_num_cols=[i for i in cars_1.columns if cars_1[i].dtype!='object']

for i in all_num_cols:

    cars_1[i]=cars_1[i].fillna(cars_1[i].mean())
code_cols=[i for i in cars_1.columns if (cars_1[i].dtype=='object')and(cars_1[i].nunique()==2)]

ohe_cols=[i for i in cars_1.columns if (cars_1[i].dtype=='object')and(cars_1[i].nunique()>2)]

all_cat_cols=[i for i in cars_1.columns if cars_1[i].dtype=='object']

st_num_cols=[i for i in cars_1.drop(['compression-ratio', 'price'], axis=1).columns if cars_1[i].dtype!='object']

log_num_cols='compression-ratio'
'''

def to_dummies(df, col):

    c=pd.get_dummies(df[col], prefix=col)

    df=df.drop(col, axis=1)

    df=pd.concat([df, c], axis=1)

    return df

def to_codes(df, col):

    df[col]=pd.Categorical(df[col]).codes

    return df

for i in code_cols:

    x=to_codes(x, i)

for i in ohe_cols:

    x=to_dummies(x,i)

'''
def cat_boost(df, cols, target):

    cb=ce.CatBoostEncoder(cols=cols)

    cb.fit(df[cols], df[target])

    c=cb.transform(df[cols]).add_suffix('_cb')

    df=df.drop(cols,axis=1)

    df=pd.concat([df, c], axis=1)

    return df

cars_1=cat_boost(cars_1, all_cat_cols, 'price')
def standard_scaler(df, col):

    df[col]=(df[col]-df[col].mean())/df[col].std()

    return df

for i in st_num_cols:

    cars_1=standard_scaler(cars_1, i)
def log_transformer(df, col):

    df[col]=np.log(df[col])

    return df

cars_1=log_transformer(cars_1, 'compression-ratio')
x=cars_1.drop('price', axis=1)

y=cars_1['price']
x_train, x_test, y_train, y_test=train_test_split(x,

                                                  y,

                                                  test_size=0.2,

                                                  random_state=42)
model=[

    {

        'name':'lasso regression',

        'estimator':Lasso(),

        'hyperparameters':{

            'alpha':np.arange(0.01, 1, 0.02)

        }

    },

    {

        'name':'decision Tree',

        'estimator':DecisionTreeRegressor(),

        'hyperparameters':{

            'max_depth':[2,3,4,5,6,7],

            'criterion':['mse', 'friedman_mse', 'mae'],

            'splitter':['best', 'random'],

            'max_features':['auto', 'sqrt', 'log2']

        }

    },

    {

        'name':'Random Forest',

        'estimator':RandomForestRegressor(),

        'hyperparameters':{

            'n_estimators':[2,3,4,5,6],

            'max_depth':[2,3,4,5,6,7],

            'max_features':['auto', 'sqrt', 'log2']

        }

    },

    {

        'name':'Extreme Gradient Boosting',

        'estimator':XGBRegressor(),

        'hyperparameters':{

            'n_etimators':[10,20,30,40,50],

            'max_depth':[2,4,6,8],

            'subsample':[0.3, 0.5, 0.7, 1],

            'learning_rate':np.arange(0.01, 0.1, 0.01)

        }

    }

]

for i in model:

    print(i['name'])

    gs=GridSearchCV(i['estimator'], param_grid=i['hyperparameters'], cv=3, n_jobs=-1, scoring='r2')

    gs.fit(x_train, y_train)

    print('best score: ', gs.best_score_)

    print('best parameters ; ', gs.best_params_)

    print('best model: ', gs.best_estimator_)

    print('\n')
xgb=XGBRegressor(learning_rate= 0.09, max_depth= 2, n_etimators= 10, subsample= 1)

xgb.fit(x_train, y_train)

preds=xgb.predict(x_test)

preds[:15]
pd.DataFrame({

    'real_values':y_test,

    'predicted_values':preds

})