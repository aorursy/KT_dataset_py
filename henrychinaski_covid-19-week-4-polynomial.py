import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm
# Load Data

train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



train.rename(columns={

        'Id': 'id',

        'Date': 'date',

        'Province_State':'state',

        'Country_Region':'country',

        'ConfirmedCases': 'confirmed',

        'Fatalities':'deaths',

        }, inplace=True)



test.rename(columns={

        'ForecastId': 'id',

        'Date': 'date',

        'Province_State':'state',

        'Country_Region':'country',

        }, inplace=True)







valid = train[train['date'] >= test['date'].min()]

train = train[(train['date'] < test['date'].min())]



valid['date'] = pd.to_datetime(valid['date'])

train['date'] = pd.to_datetime(train['date'])

test['date'] = pd.to_datetime(test['date'])



train['date'] = (train['date'] - pd.Timestamp('2020-03-01')).dt.days

valid['date'] = (valid['date'] - pd.Timestamp('2020-03-01')).dt.days

test['date']  = (test['date'] - pd.Timestamp('2020-03-01')).dt.days



train['loc'] = train['country'].astype(str) + '-' + train['state'].astype(str)

valid['loc'] = valid['country'].astype(str) + '-' + valid['state'].astype(str)

test['loc'] = test['country'].astype(str) + '-' + test['state'].astype(str)
def get_order(max_order):

    RMSE_order={} #Dict for the key:value pairs of order and RMSE

    for order in range(1,max_order,1):    #the maximum order for the optimization is set here

        z = np.polyfit(X_train_.values, y_train_.values, order)

        pf = np.poly1d(z)

        

        y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)



        predictions[coords] = y_preds_



        RMSE_get_order=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

        RMSE_order[order]=RMSE_get_order

        result = min(RMSE_order, key=RMSE_order.get)

    return result
all_coords = train['loc'].unique().tolist()

print(all_coords)
predictions = dict()

RMSE = dict()

total_RMSE = 0

fit_order=1 # If get_order fails the fit_order is set to

all_orders_used=[]

#This is for visualising the data. As there are 306 datasets only a fraction of 20 sets can be choosen here

_, ax = plt.subplots(10,2, figsize=(15, 50))

ax = ax.flatten()



for k,coords in tqdm(enumerate(all_coords[70:90])):   #Define the part of the dataset you want to look at, e.g. [200:220]

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['loc']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    fit_order=get_order(11) # Here the order up to the maximum order is optimized to the data

    all_orders_used.append(fit_order)

    

    z = np.polyfit(X_train_.values, y_train_.values, fit_order)

    pf = np.poly1d(z)

        

    y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)

    

    predictions[coords] = y_preds_

    RMSE[coords]=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

    total_RMSE += np.sqrt(np.sum(np.square(y_preds_-y_valid_)))



    sns.lineplot(x=valid[valid['loc']==coords]['date'], y=valid[valid['loc']==coords]['confirmed'], label='y-valid',ax=ax[k])

    sns.lineplot(x=train[train['loc']==coords]['date'], y=train[train['loc']==coords]['confirmed'], label='y-train',ax=ax[k])

    sns.lineplot(x=valid[valid['loc']==coords]['date'], y=y_preds_, label='y-preds',ax=ax[k])

    ax[k].set_title(f'Confirmed cases: ({coords})')





print(total_RMSE)

print(all_orders_used)
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0

fit_order=1 # If get_order fails the fit_order is set to

all_orders_used=[]



#This is for visualising the data. As there are 306 datasets only a fraction of 20 sets can be choosen here

_, ax = plt.subplots(10,2, figsize=(15, 50))

ax = ax.flatten()



for k,coords in tqdm(enumerate(all_coords[260:280])): #Define the part of the dataset you want to look at

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['loc']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    fit_order=get_order(11)

    all_orders_used.append(fit_order)



    z = np.polyfit(X_train_.values, y_train_.values, fit_order)

    pf = np.poly1d(z)

        

    y_preds_ = np.round(X_valid_.apply(pf)).clip(lower=y_linear)

    

    predictions[coords] = y_preds_

    RMSE[coords]=np.sqrt(np.sum(np.square(y_preds_-y_valid_)))

    total_RMSE += np.sqrt(np.sum(np.square(y_preds_-y_valid_)))



    sns.lineplot(x=valid[valid['loc']==coords]['date'], y=valid[valid['loc']==coords]['deaths'], label='y-valid',ax=ax[k])

    sns.lineplot(x=train[train['loc']==coords]['date'], y=train[train['loc']==coords]['deaths'], label='y-train',ax=ax[k])

    sns.lineplot(x=valid[valid['loc']==coords]['date'], y=y_preds_, label='y-preds',ax=ax[k])

    ax[k].set_title(f'Fatalities: ({coords})')

    

print(total_RMSE)

print(all_orders_used)
submission = pd.DataFrame()

submission['loc'] = test['loc']

submission.reset_index(inplace=True)



submission['ConfirmedCases'] = 0

submission['Fatalities'] = 0
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0

fit_order=1



for coords in all_coords:

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['loc']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['loc']==coords]['confirmed']#.values.reshape(-1,1)

    

    X_test_ = test[test['loc']==coords]['date']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    fit_order=get_order(11)

    

    z = np.polyfit(X_train_.values, y_train_.values, fit_order)

    pf = np.poly1d(z)

    

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_test_)+1,1)

        

    y_preds_ = np.round(X_test_.apply(pf)).clip(lower=y_linear)

    



    submission.loc[submission['loc']==coords,'ConfirmedCases'] = y_preds_
all_coords = train['loc'].unique().tolist()

predictions = dict()

RMSE = dict()

total_RMSE = 0



fit_grade=1



for coords in all_coords:

    

    X_train_ = train[train['loc']==coords]['date']#.values.reshape(-1,1)

    y_train_ = train[train['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    X_valid_ = valid[valid['loc']==coords]['date']#.values.reshape(-1,1)

    y_valid_ = valid[valid['loc']==coords]['deaths']#.values.reshape(-1,1)

    

    X_test_ = test[test['loc']==coords]['date']#.values.reshape(-1,1)

    

    last_diff = y_train_.iloc[-1] - y_train_.iloc[-2]

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_valid_)+1,1)

    

    fit_order=get_order(11)



    z = np.polyfit(X_train_.values, y_train_.values, fit_order)

    pf = np.poly1d(z)

        

    y_linear = y_train_.iloc[-1] + last_diff*np.arange(1,len(X_test_)+1,1)

    

    y_preds_ = np.round(X_test_.apply(pf)).clip(lower=y_linear)

    



    

    submission.loc[submission['loc']==coords,'Fatalities'] = y_preds_
submission.drop('loc', axis=1, inplace=True)

submission['index'] = submission['index'] + 1

submission.rename(columns={

    'index' : 'ForecastId'}, inplace=True)
submission
submission.to_csv('submission.csv', index=False)