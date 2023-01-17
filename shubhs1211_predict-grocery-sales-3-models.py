import numpy as np

import pandas as pd

import random

from sklearn import preprocessing, metrics

import gc; gc.enable()

import random



from sklearn.neural_network import MLPRegressor



from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



np.random.seed(100)
print('Three models: \n1. Neural network \n2. Bagging\n3. Random Forest')
# read datasets

datatypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':str}

data = {

    'traind': pd.read_csv('../input/train.csv', dtype=datatypes, parse_dates=['date']),

    'testd': pd.read_csv('../input/test.csv', dtype=datatypes, parse_dates=['date']),

    'itemsd': pd.read_csv('../input/items.csv'),

    'storesd': pd.read_csv('../input/stores.csv'),

    'transactionsd': pd.read_csv('../input/transactions.csv', parse_dates=['date']),

    'holidaysd': pd.read_csv('../input/holidays_events.csv', dtype={'transferred':str}, parse_dates=['date']),

    'oild': pd.read_csv('../input/oil.csv', parse_dates=['date']),

    }

# dataset preprocessing

print('Datasets pre-processing')



train = data['traind'][(data['traind']['date'].dt.month == 8) & (data['traind']['date'].dt.day > 15)]

del data['traind']; gc.collect();

target = train['unit_sales'].values

target[target < 0.] = 0.

train['unit_sales'] = np.log1p(target)



def label_encoding(df):

    for c in df.columns:

        if df[c].dtype == 'object':

            lbl = preprocessing.LabelEncoder()

            df[c] = lbl.fit_transform(df[c])

            print(c)

    return df
def transform_dataframe(df):

    df['date'] = pd.to_datetime(df['date'])

    df['yea'] = df['date'].dt.year

    df['mon'] = df['date'].dt.month

    df['day'] = df['date'].dt.day

    df['date'] = df['date'].dt.dayofweek

    df['onpromotion'] = df['onpromotion'].map({'False': 0, 'True': 1})

    df['perishable'] = df['perishable'].map({0:1.0, 1:1.25})

    df = df.fillna(-1)

    return df
data['itemsd'] = label_encoding(data['itemsd'])

train = pd.merge(train, data['itemsd'], how='left', on=['item_nbr'])

test = pd.merge(data['testd'], data['itemsd'], how='left', on=['item_nbr'])

del data['testd']; gc.collect();

del data['itemsd']; gc.collect();
train = pd.merge(train, data['transactionsd'], how='left', on=['date','store_nbr'])

test = pd.merge(test, data['transactionsd'], how='left', on=['date','store_nbr'])

del data['transactionsd']; gc.collect();

target = train['transactions'].values

target[target < 0.] = 0.

train['transactions'] = np.log1p(target)
data['storesd'] = label_encoding(data['storesd'])

train = pd.merge(train, data['storesd'], how='left', on=['store_nbr'])

test = pd.merge(test, data['storesd'], how='left', on=['store_nbr'])

del data['storesd']; gc.collect();



data['holidaysd'] = data['holidaysd'][data['holidaysd']['locale'] == 'National'][['date','transferred']]

data['holidaysd']['transferred'] = data['holidaysd']['transferred'].map({'False': 0, 'True': 1})

train = pd.merge(train, data['holidaysd'], how='left', on=['date'])

test = pd.merge(test, data['holidaysd'], how='left', on=['date'])

del data['holidaysd']; gc.collect();
train = pd.merge(train, data['oild'], how='left', on=['date'])

test = pd.merge(test, data['oild'], how='left', on=['date'])

del data['oild']; gc.collect();



train = transform_dataframe(train)

test = transform_dataframe(test)

col = [c for c in train if c not in ['id', 'unit_sales','perishable','transactions']]
x1 = train[(train['yea'] != 2016)]

x2 = train[(train['yea'] == 2016)]

del train; gc.collect();



y1 = x1['transactions'].values

y2 = x2['transactions'].values
def NWRMSLE(y, pred, w):

    return metrics.mean_squared_error(y, pred, sample_weight=w)**0.5


print('\nRunning the models')    



no_of_models = 3



for model in range(1, no_of_models + 1):

    

    

    random_state1 = round(model + 515 * model + 56 * model) 

    np.random.seed(random_state1)

    

    print('\nmodel = ', model)

    

    if (model == 1):

        print('Neural Network')

        model_name = 'Neural Network'    

        clf = MLPRegressor(hidden_layer_sizes=(4,), max_iter=30) 

    

    if (model == 2):

        print('Bagging Regressor')

        model_name = 'BaggingRegressor'

        clf = BaggingRegressor(

                DecisionTreeRegressor(

                        max_depth=6,

                        max_features=0.85))  

        

    if (model == 3):

        print('Random forest')

        model_name = 'RandomForest'

        clf = RandomForestRegressor(n_estimators=70, max_depth = 3, n_jobs = -1, 

                                   random_state=random_state1, verbose=0, warm_start=True)

                                          

    clf.fit(x1[col], y1)





    random1 = NWRMSLE(y2, clf.predict(x2[col]), x2['perishable'])

    

    

    test['transactions'] = clf.predict(test[col])

    test['transactions'] = test['transactions'].clip(lower = 0.+1e-12)



    col = [c for c in x1 if c not in ['id', 'unit_sales','perishable']]

    y1 = x1['unit_sales'].values

    y2 = x2['unit_sales'].values

    

    random_state2 = round(model + 331 * model + 561 * model) 

    np.random.seed(random_state2)



    if (model == 1):

        clf = MLPRegressor(hidden_layer_sizes = (4,), max_iter = 30)



    if (model == 2):

        clf = BaggingRegressor(DecisionTreeRegressor(max_depth = 5,max_features = 0.85))            



    if (model == 3):

        clf = RandomForestRegressor(n_estimators = 70, max_depth = 3, n_jobs = -1, 

                                   random_state=random_state2, verbose=0, warm_start=True)

        

    clf.fit(x1[col], y1)

    

    random2 = NWRMSLE(y2, clf.predict(x2[col]), x2['perishable'])

   

    print('Performance: NWRMSLE(1) = ',random1,'NWRMSLE(2) = ',random2)



    test['unit_sales'] = clf.predict(test[col])

    cut = 0. + 1e-12 

    

    test['unit_sales'] = (np.exp(test['unit_sales']) - 1).clip(lower = cut)





    output_file = 'submission ' + str(model_name) + '.csv'

 

    test[['id','unit_sales']].to_csv(output_file, index=False, float_format='%.2f')



        

        

        