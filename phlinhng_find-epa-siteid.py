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
station=pd.read_csv("../input/air_station.csv")
station.head(20)
air2017=pd.read_pickle("../input/aqi_tw_2017.pkl")
list(air2017)
air2017_taipei=air2017[air2017['COUNTYID']==63000] #63000 Taipei City
air2017_taipei.head(10)
# select a specific range of data
mask = (air2017_taipei["UPDATETIME"] >= "2017-01-02 00:00") & (air2017_taipei["UPDATETIME"] < "2017-01-02 23:59")
air2017_taipei[mask][air2017_taipei[mask]['CO'] == 0.51]
air2017_taichung=air2017[air2017['COUNTYID']==66000] #66000 Taichung City
mask = (air2017_taichung["UPDATETIME"] >= "2017-01-02 00:00") & (air2017_taichung["UPDATETIME"] < "2017-01-02 23:59")
air2017_taichung[mask][air2017_taichung[mask]['CO'] == 0.69]