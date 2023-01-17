# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from pandas.io.json import json_normalize
JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
train = pd.read_csv('../input/train.csv', converters={column:json.loads for column in JSON_COLUMNS}, 
                   dtype={'fullVisitorId': 'str'})
train.info()
pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
for column in JSON_COLUMNS:
    columns_df = json_normalize(train[column])
    columns_df.columns = [f"{column}_{subcolumn}" for subcolumn in columns_df.columns]
    train = train.drop(column, axis=1).merge(columns_df, right_index=True, left_index=True)
#     print(columns_df.head())
train.head()
