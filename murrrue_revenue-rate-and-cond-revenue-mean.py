# thx for import: kernel by Julián Peller

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

df_train = load_df()
df_test = load_df("../input/test.csv")
# compute rate and conditional revenue on test data:
df_train['totals.transactionRevenue']=df_train['totals.transactionRevenue'].astype(float)
train=pd.DataFrame()
train['target']=np.log1p(df_train.groupby('fullVisitorId')['totals.transactionRevenue'].sum())
print('Explicit')
print('Cases: '+str((train.target>0).mean())+', Conditional Mean: '+str(train[train.target>0].target.mean()))
# compute rate and conditional mean through rmse:
n=len(train)
y0=np.round(np.sqrt(((train.target-0)**2).mean()),4)
y1=np.round(np.sqrt(((train.target-1)**2).mean()),4)
a=2*y0**2/(y0**2-y1**2+1)
m=n*y0**2/a**2
print('Implicit')
print('Cases: '+str(m/n)+', Conditional Mean: '+str(a))
# get leaderboard rmse for x=1
# submission=pd.read_csv('../input/sample_submission.csv',index_col=[0])
# submission['PredictedLogRevenue']=1
# submission.to_csv('submission1.csv')
# this actually returns a leaderboard rmse of 1.9529
n=0.3*len(df_test)
y0=1.7804
y1=1.9529
a=2*y0**2/(y0**2-y1**2+1)
m=n*y0**2/a**2
print('Implicit on Public Leaderboard')
print('Cases: '+str(m/n)+', Conditional Mean: '+str(a))