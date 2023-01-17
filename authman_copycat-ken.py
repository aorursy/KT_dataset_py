import numpy as np

import pandas as pd

import gc



tr = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

te = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')



a = set(tr.building_id)

b = set(te.building_id)

len(a-b), len(b-a)
del a,b; gc.collect()

tr.timestamp.min(), tr.timestamp.max()
tr.timestamp = tr.timestamp.map(lambda x: x[5:])

te.timestamp = te.timestamp.map(lambda x: x[5:])

te = te.merge(

    tr[['building_id','meter','timestamp','meter_reading']],

    how='left',

    on=['building_id','meter','timestamp']

)
del tr; gc.collect()

(te.meter_reading.isna().sum() / te.shape[0] * 100).astype(np.int)
fillna = te.groupby(['building_id', 'meter']).meter_reading.mean().reset_index()

fillna.rename(columns={'meter_reading':'missing'}, inplace=True)

te = te.merge(fillna, how='left', on=['building_id', 'meter'])



mask = te.meter_reading.isna()

te.loc[mask, 'meter_reading'] = te[mask].missing

(te.meter_reading.isna().sum() / te.shape[0] * 100).astype(np.int)
te[['row_id', 'meter_reading']].to_csv('./submission.csv', index=False)