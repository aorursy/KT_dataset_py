import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline

from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import os
print(os.listdir("../input"))
# Import all of them 
sales=pd.read_csv("../input/sales_train.csv")

# settings
import warnings
warnings.filterwarnings("ignore")

item_cat=pd.read_csv("../input/item_categories.csv")
item=pd.read_csv("../input/items.csv")
sub=pd.read_csv("../input/sample_submission.csv")
shops=pd.read_csv("../input/shops.csv")
test=pd.read_csv("../input/test.csv")
shops.describe().T
N_SESSIONS = 34
# Now we convert the raw sales data to monthly sales, broken out by item & shop
sales_piv= sales.pivot_table(index=['item_id','shop_id'], columns='date_block_num',values='item_cnt_day',aggfunc=np.sum,fill_value=0).reset_index()
sales_piv.head()
# Merge the monthly sales data to the test data
Test=pd.merge(test, item, how='inner', on='item_id')
Test = pd.merge(Test, sales_piv, on=['item_id','shop_id'], how='left').fillna(0)
#add category
Testcat=Test['item_category_id']
Test = Test.drop(labels=['ID', 'shop_id', 'item_id','item_name','item_category_id'], axis=1).T
# times series ARIMA
Ttrain=Test.append(Test.diff(1).dropna())
Ttrain=Ttrain.append(pd.rolling_mean(Test,12)).dropna()


Ttrain=Ttrain.T
#cluster in 9 groups
from sklearn.cluster import KMeans
clu=KMeans(n_clusters=9)
Ttrain['clu']=clu.fit_predict(Ttrain)
Ttrain['cat']=Testcat

Ttrain
ALPHA = 0.5
BETA = 0.0

def create_filtered_prediction(train_ts, alpha, beta):
    train_time_filtered_ts = np.zeros((train_ts.shape[0], N_SESSIONS), dtype=np.float)
    train_time_filtered_ts[0, :] = train_ts[0, :N_SESSIONS]
    train_memontum_ts = np.zeros((train_ts.shape[0], N_SESSIONS), dtype=np.float)
    prediction_ts = np.zeros((train_ts.shape[0], N_SESSIONS+1), dtype=np.float)
    for i in range(1, N_SESSIONS):
        train_time_filtered_ts[:, i] = (1-alpha) * (train_time_filtered_ts[:, i-1] + \
                                                    train_memontum_ts[:, i-1]) + alpha * train_ts[:, i]
        train_memontum_ts[:, i] = (1-beta) * train_memontum_ts[:, i-1] + \
                                  beta * (train_time_filtered_ts[:, i] - train_time_filtered_ts[:, i-1])
        prediction_ts[:, i+1] = train_time_filtered_ts[:, i] + train_memontum_ts[:, i]
    return prediction_ts

#predictions = create_filtered_prediction(Ttrain.values, ALPHA, BETA)
N_SESSIONS=32
month=33
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print(Ttrain.shape)
yo=Ttrain.iloc[:,month]
yo.colomns=[month,'di','ma']
#print(yo)
for Ai in range(28,29,1):
    for Bi in range(59,61,1):
        BETA=Bi/100
        ALPHA=Ai/100
        p= create_filtered_prediction(Ttrain.values, ALPHA, BETA)
        print(p[:,N_SESSIONS])
        error = rmse(yo[month], p[:,N_SESSIONS])
        print('alpha %d beta %d month %d - Error %.2f' % (Ai,Bi,N_SESSIONS, error) )
          
#predictions
        N_SESSIONS=33
        p= create_filtered_prediction(Ttrain.values, 0.27, 0.63)
        print(p[:,33])
        error = rmse(yo[month], p[:,33])
        print('alpha %d beta %d month %d - Error %.2f' % (Ai,Bi,33, error) )
        N_SESSIONS=33
        p= create_filtered_prediction(Ttrain.values, 0.5, 0.0)
        print(p[:,33])
        error = rmse(yo[month], p[:,33])
        print('alpha %d beta %d month %d - Error %.2f' % (Ai,Bi,33, error) )
sub["item_cnt_month"] = np.clip(p[:, N_SESSIONS], 0, 20)
sub.to_csv("submitalpha25bet057.csv",index=False)
                  #.format(ALPHA))