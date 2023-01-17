import numpy as np
import pandas as pd
columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        }
training = pd.read_csv( "../input/train.csv", 
                        nrows=122071523, 
                        usecols=columns, 
                        dtype=dtypes)
training.tail()
valid1 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,144708153), 
                      nrows=7705357, 
                      usecols=columns, 
                      dtype=dtypes)
valid1.head()
valid1.tail()
valid2 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,161974466), 
                      nrows=6291379, 
                      usecols=columns, 
                      dtype=dtypes)
valid2.head()
valid2.tail()
valid2 = pd.concat([valid1, valid2])
valid2.head()
valid2.tail()
del valid1
import gc
gc.collect()
valid3 = pd.read_csv( "../input/train.csv", 
                      skiprows=range(1,174976527), 
                      nrows=6901686, 
                      usecols=columns, 
                      dtype=dtypes)
valid3.head()
valid3.tail()
valid3 = pd.concat([valid2,valid3])
valid3.head()
valid3.tail()
del valid2
gc.collect()
validation = valid3
del valid3
gc.collect()
validation.head()
validation.tail()
