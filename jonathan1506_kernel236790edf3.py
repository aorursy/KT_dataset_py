# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def date_to_timestamp(s):
  return time.mktime(datetime.strptime(s, '%d-%m-%Y').timetuple())
from datetime import datetime
import time
# 1577836800 <- Wednesday 1 January 2020
# 1578182400 <- Sunday 5 January 2020
# 604800.0 <- 1 week
# 86400.0 <- 1 day
def convert_date(s):
    try:
        return datetime.fromtimestamp(s).strftime('%d-%m-%Y')
    except:
        return s

def convert_address(s):
    cities = ["manila", "luzon", "visayas", "mindanao"]
    for i, city in enumerate(cities):
        if city == s.split()[-1].lower():
            return i
    else:
        return -1
filename = r'/kaggle/input/logistics-shopee-code-league/delivery_orders_march.csv'
data = pd.read_csv(filename)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
data["buyeraddress"].apply(convert_address).value_counts()
data["1st_deliver_attempt"] = data["1st_deliver_attempt"].apply(convert_date)
data["2nd_deliver_attempt"] = data["2nd_deliver_attempt"].apply(convert_date)
data["buyeraddress"] = data["buyeraddress"].apply(convert_address)
data["selleraddress"] = data["selleraddress"].apply(convert_address)
data["pick"] = data["pick"].apply(convert_date)
data["buyeraddress"].value_counts()
np.busday_count(data.head(1)["pick"], data.head(1)["pick"])
data.head(1)["pick"] - data.head(1)["1st_deliver_attempt"]
data.head(2)