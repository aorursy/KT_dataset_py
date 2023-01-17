import pandas as pd

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

train['timestamp'] = train['timestamp'].str[5:]

test['timestamp'] = test['timestamp'].str[5:]

submission = test.merge(train, how='left', on =['building_id','timestamp','meter'])

submission[['row_id','meter_reading']].fillna(0).to_csv('submission.csv',index=False)