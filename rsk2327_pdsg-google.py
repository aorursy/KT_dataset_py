import os
import numpy as np
import pandas as pd

from pandas.io.json import json_normalize
import json
import matplotlib.pyplot as plt
from sklearn import preprocessing

import datetime


%matplotlib inline
train = pd.read_csv('../input/train_final.csv')
test = pd.read_csv('../input/test_final.csv')
train.head()
train.channelGrouping.dtype
train.channelGrouping.value_counts()
#Unique values
len(train.channelGrouping.value_counts())
#Missing Percentage
sum(pd.isna(train.channelGrouping))/len(train)
train.trafficSource_source.value_counts()
len(train.trafficSource_source.value_counts())
380
sum(pd.isna(train.trafficSource_source))/len(train)
train['trafficSource_adwordsClickInfo.adNetworkType'].value_counts()
sum(pd.isna(train['trafficSource_adwordsClickInfo.adNetworkType']))/len(train)






