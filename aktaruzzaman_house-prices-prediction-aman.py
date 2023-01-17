#Import required libs

import pandas as pd

import numpy as np

import seaborn as sns;sns.set()

import matplotlib.pyplot as plt



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
#Import raw data and copy them in test and train variable. Concat them together for feature engineering

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

t = train.copy()

ts = test.copy()

d = pd.concat([t.drop('SalePrice',axis=1),ts])
#Missing value

miss_col = t.columns[1:-1]

def fill_na(df):

    for col in miss_col:

        if (df[col].dtype =='int64'):

            df[col]=df[col].fillna(df[col].mode()[0])

        elif (df[col].dtype =='float64'):

            df[col]=df[col].fillna(df[col].mode()[0])

        else:

            df[col]=df[col].fillna('N')

    return df
#continious var -> Catagorical

cut_cols=t.columns[1:-1]

def col_cut(df):

    for col in cut_cols:

        if ((df[col].dtype=='int64')|(df[col].dtype=='float64')):

            df[col]=pd.cut(df[col],30)

    return df
t.dtypes.value_counts()
# fill missing value + continious_var to Catagorical_var

#df = col_cut(df)

def clean(df):

    df = fill_na(df)

    return df
t = clean(t)

ts = clean(ts)

d = pd.concat([t.drop('SalePrice',axis=1),ts])
# Lebel Encoding function

from sklearn import preprocessing

lebel_cols = t.columns[1:-1]



def lebel_encoding(df):

    for col in lebel_cols:

        if (df[col].dtype=='object'):

            le = preprocessing.LabelEncoder()

            le = le.fit(d[col])

            df[col] = le.transform(df[col])

    return df
# Lebel Encoding execution

t = lebel_encoding(t)

ts = lebel_encoding(ts)
# X and Y for train model Y = wX + b

X = t.drop(['Id','SalePrice'],axis=1)

Y = t.SalePrice



ts_sub = ts.drop('Id',axis=1)
# Model selection

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
#Split X and Y in test and evaluate set

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
# Define error matrics and tune model parameter

error = mean_absolute_error

rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

           max_features='auto', max_leaf_nodes=481,

           min_impurity_decrease=0.0, min_impurity_split=None,

           min_samples_leaf=1, min_samples_split=2,

           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=None,

           oob_score=False, random_state=1, verbose=0, warm_start=False)

rf = rf.fit(X_train,Y_train)

Y_SalePrice = rf.predict(X_test)

error = error(Y_SalePrice,Y_test)

error
# Fit model and predict sales price

rf = rf.fit(X,Y)

Sale_Price_sub = rf.predict(ts_sub) 
ts['SalePrice'] = Sale_Price_sub

ts.columns
#Save result for submission

ts[['Id','SalePrice']].to_csv('Sub1.csv',index=False)