import numpy as np 

import pandas as pd



train = pd.read_csv("../input/data_set_ALL_AML_train.csv")

test = pd.read_csv("../input/data_set_ALL_AML_independent.csv")

actual = pd.read_csv("../input/actual.csv")
train.head() #First 5 rows
train.shape
train.columns
train.describe()
#remove  some string columns from data

drop_elements = ['Gene Description','call', 'call.1',

       'call.2',  'call.3',  'call.4', 'call.5',

       'call.6', 'call.7','call.8',  'call.9', 'call.10',

        'call.11', 'call.12',  'call.13',  'call.14',

        'call.15', 'call.16', 'call.17', 'call.18',

        'call.19',  'call.20', 'call.21',  'call.22',

        'call.23',  'call.24', 'call.25', 'call.26'

       , 'call.27', 'call.28', 'call.29', 'call.30',

        'call.31',  'call.32', 'call.33', 'call.34'

       , 'call.35', 'call.36', 'call.37']

trainD = train.drop(drop_elements, axis=1)

trainD.head()

trainD = trainD.T
