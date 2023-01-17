from datetime import date, timedelta
import gc # garbage collector
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

import os
print(os.listdir("../input/favorita-grocery-sales-forecasting"))
!pip install patool
!pip install pyunpack
import pyunpack

def unzip(zip_path, data_folder):
    print('Unzipping file: {}'.format(zip_path))
    pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
#       if not os.path.exists(output_file):
#         raise ValueError(
#             'Error in unzipping process! {} not found.'.format(output_file))



list_dir =  os.listdir("../input/favorita-grocery-sales-forecasting")
for i in list_dir:
    unzip("../input/favorita-grocery-sales-forecasting/"+i, '../input/favorita-grocery-sales-forecasting')
list_dir =  os.listdir("../input/favorita-grocery-sales-forecasting")
list_dir

from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import lightgbm as lgb

df_train = pd.read_csv(
    '../input/train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)

df_test = pd.read_csv(
    "../input/test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "../input/items.csv",
).set_index("item_nbr")

stores = pd.read_csv(
    "../input/stores.csv",
).set_index("store_nbr")

le = LabelEncoder()
items['family'] = le.fit_transform(items['family'].values)

stores['city'] = le.fit_transform(stores['city'].values)
stores['state'] = le.fit_transform(stores['state'].values)
stores['type'] = le.fit_transform(stores['type'].values)

df_2017 = df_train.loc[df_train.date>=pd.datetime(2017,1,1)]
