# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import json
import os
from pandas.io.json import json_normalize
def read_data(path,nrows):
    df = pd.read_csv(path,
                converters = {column: json.loads for column in ["device","geoNetwork","totals","trafficSource"]},
                 dtype = {"fullVisitorId" : "str"},
                 nrows=nrows)
    for column in ["device","geoNetwork","totals","trafficSource"]:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

df_small = pd.DataFrame(read_data("/kaggle/input/ga-customer-revenue-prediction/train.csv",200000))
for column in df_small.columns:
    if len(df_small[column].unique()) == 1:
        df_small.drop(column,axis=1,inplace=True)
df_small.drop(["device.isMobile","geoNetwork.continent",'geoNetwork.metro','visitId',
               'sessionId','trafficSource.referralPath','trafficSource.adwordsClickInfo.gclId',
               'trafficSource.campaignCode'],axis=1,inplace=True)
mapeo=dict()
continents = list()
for i,name in enumerate(df_small["geoNetwork.subContinent"].unique()):
    continents.append(pd.DataFrame(df_small.loc[df_small["geoNetwork.subContinent"] == name, : ]))
    mapeo[i] = name
df_small['trafficSource.adwordsClickInfo.slot'].value_counts()