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
# import our bq_helper package
import bq_helper
# create a helper object for bigquery dataset
open_aq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                   dataset_name = "openaq")
open_aq.list_tables()

#open_aq.head("global_air_quality")

query = """SELECT *
           FROM `bigquery-public-data.openaq.global_air_quality`
        """
all_data = open_aq.query_to_pandas(query)
all_data.to_csv("all_pollution.csv")
# preprocessing data... 
# 全部約有20000筆資料 使用過濾
"""
Unnamed: 0                25398
location                   7757
city                       2386
country                      68
pollutant                     7
value                      5100
timestamp                  1457
unit                          2
source_name                 112
latitude                   7528
longitude                  7591
averaged_over_in_hours        9
"""
# list出一個column中的different值，組成一個list再每項丟進去groupby.count中
"""['US' 'FR' 'CA' 'BA' 'TH' 'IN' 'NL' 'GB' 'CH' 'TR' 'PL' 'PT' 'ES' 'BR'
 'PE' 'NO' 'HK' 'LV' 'IT' 'CN' 'GH' 'CL' 'CO' 'SI' 'BD' 'AE' 'MT' 'VN'
 'BE' 'TW' 'LT' 'KZ' 'DE' 'SE' 'NG' 'MK' 'AU' 'BH' 'PH' 'RU' 'AD' 'IE'
 'AT' 'ID' 'HU' 'LK' 'CW' 'UG' 'FI' 'KE' 'IL' 'LU' 'HR' 'XK' 'UZ' 'GI'
 'NP' 'SG' 'DK' 'CZ' 'MX' 'ET' 'KW' 'MN' 'AR' 'RS' 'ZA' 'SK']"""
list = all_data.country.unique()
print(list), print(len(list))

a = all_data['country'].value_counts() # 知道每個國家有多少資料，並會sort

# 因為CN最多，因此只將country=CN取出來
CN_pollutant = all_data.loc[all_data['country'] == "CN"]
print(CN_pollutant)

# 再將CN的污染拆分成不同的pollutant(pm25, o3, so2...)
list = CN_pollutant.pollutant.unique()
print(list) # 類似['o3' 'co' 'pm10' 'so2' 'pm25' 'no2']
a = CN_pollutant['pollutant'].value_counts() # 6種污染物的資料量幾乎一樣

# initial each type of pollutant as a dict key and map to a dataframe
d = dict.fromkeys(list, "dataframe")
print(d)
for key in d:
    d[key] = CN_pollutant.loc[CN_pollutant['pollutant'] == key]
# use "d"(data) 以station為單位，產生timeseries
for key in d:
    print(key)
    print(d[key].shape)
