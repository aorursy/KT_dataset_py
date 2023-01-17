import numpy as np

import os

import pandas as pd



PATH = "../input"



dtypes = {

          'id' : np.dtype('int64'),

          'store_nbr': np.dtype('int64'),

          'item_nbr': np.dtype('int64'),

          'unit_sales': np.dtype('float64'),

          'onpromotion': np.dtype('O')}



def load_train_data(data_path=PATH, file_name="train.csv", condition=None, dtypes=None):

    df = pd.DataFrame( )

    csv_path = os.path.join(data_path, file_name)

    reader = pd.read_csv(csv_path, parse_dates=['date'], dtype=dtypes , iterator=True,  chunksize=10000000)

    return pd.concat([chunk.query(condition) for chunk in reader])

   

        

train = load_train_data(data_path=PATH, file_name="train.csv", condition="item_nbr == 103665", dtypes=dtypes)



train.shape
train.head()
import gc

gc.collect()