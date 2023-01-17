# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc; gc.enable()

from sklearn import preprocessing, linear_model, metrics

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

data = {

    'tra': pd.read_csv('../input/train.csv', dtype=dtypes, parse_dates=['date']),

    'tes': pd.read_csv('../input/test.csv', dtype=dtypes, parse_dates=['date']),

    'ite': pd.read_csv('../input/items.csv'),

    'sto': pd.read_csv('../input/stores.csv'),

    'trn': pd.read_csv('../input/transactions.csv', parse_dates=['date']),

    'hol': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    'oil': pd.read_csv('../input/oil.csv', parse_dates=['date']),

    }
train = data['tra'][(data['tra']['date'].dt.month == 8) & (data['tra']['date'].dt.day > 15)]

test = data['tes'][(data['tes']['date'].dt.month == 8) & (data['tes']['date'].dt.day > 15)]
#train["item_nbr"].value_counts(sort = True,ascending=False).nlargest(5)
#strain_502331= train[(train["item_nbr"] == 502331)]

#= test[(test["item_nbr"] == 502331)]
strain = train #strain_502331

stest = test #stest_502331

print(strain.shape,stest.shape)
target = strain['unit_sales'].values

target[target < 0.] = 0.

strain['unit_sales'] = target
strain = pd.merge(strain, data['ite'], how='left', on=['item_nbr'])

strain = pd.merge(strain, data['sto'], how='left', on=['store_nbr'])

data_h_1 = data['hol'][data['hol']['locale'] == 'National'][['date','transferred']]

data_h_1['transferred'] = data_h_1['transferred'].map({'False': 0, 'True': 1})

strain = pd.merge(strain, data_h_1, how='left', on=['date'])

strain = pd.merge(strain, data['oil'], how='left', on=['date'])



stest = pd.merge(stest, data['ite'], how='left', on=['item_nbr'])

stest = pd.merge(stest, data['sto'], how='left', on=['store_nbr'])

data_h_t = data['hol'][data['hol']['locale'] == 'National'][['date','transferred']]

data_h_t['transferred'] = data_h_t['transferred'].map({'False': 0, 'True': 1})

stest = pd.merge(stest, data_h_t, how='left', on=['date'])

stest = pd.merge(stest, data['oil'], how='left', on=['date'])
from sklearn import preprocessing

def df_transform(df):

    df['date'] = pd.to_datetime(df['date'])

    df['yea'] = df['date'].dt.year

    df['mon'] = df['date'].dt.month

    df['day'] = df['date'].dt.day

    df['dayofweek'] = df['date'].dt.dayofweek

    df['onpromotion'] = df['onpromotion'].map({'False': 1, 'True': 2})

    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})

    df = df.fillna(0)

    return df

def df_lbl_enc(df):

    for c in df.columns:

        if df[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            df[c] = lbl.fit_transform(df[c])

            print(c)

    return df

strain_t = df_transform(strain)

strain_t_e = df_lbl_enc(strain_t)



stest_t = df_transform(stest)

stest_t_e = df_lbl_enc(stest_t)
strain_t_e_dateIndex = strain_t_e.set_index('date')

stest_t_e_dateIndex = stest_t_e.set_index('date')



col =[c for c in strain_t_e_dateIndex if c not in ['id','item_nbr','mon','class','city','cluster','unit_sales']]

train_features = strain_t_e_dateIndex[col]

target = np.log1p(strain_t_e_dateIndex[['unit_sales']])



col =[c for c in stest_t_e_dateIndex if c not in ['id','item_nbr','mon','class','city','cluster']]

features = stest_t_e_dateIndex[col]
from sklearn.model_selection import train_test_split





#X_train = train_features['2015':'2016']

#X_test = train_features['2013']

#y_train = target['2014':'2017']

#y_test = target['2013']



X_train, X_test, y_train, y_test = train_test_split(train_features, target, test_size=0.20, random_state=42)
W_train = X_train['perishable']#.map({0:1.0, 1:1.25})

W_test = X_test['perishable']#.map({0:1.0, 1:1.25})

#W_train = W_train.fillna(0)

#W_test = W_test.fillna(0)
from sklearn import metrics

def NWRMSLE(y, pred, w):

    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import cross_validation



rf = RandomForestRegressor(max_features = "auto", random_state =50 )



rf.fit(X_train, y_train)

print ('RF accuracy: TRAINING', rf.score(X_train,y_train,W_train))

print ('RF accuracy: TESTING', rf.score(X_test,y_test,W_test))

print("feature Importance",rf.feature_importances_)

yhat1 = rf.predict(X_test)

print('NWRMSLE RF',NWRMSLE((y_test),(yhat1),W_test.values ))
test.head()
#y_test['pred']=yhat1

#plt.plot(y_test['unit_sales'])

#plt.plot(y_test['pred'])

#plt.show()

#Submission

test['unit_sales'] = rf.predict(features[col])

test[['id','unit_sales']].to_csv("submission_rf.csv", index=False, float_format='%.2f')