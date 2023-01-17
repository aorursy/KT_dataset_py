# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import time,sys,datetime
import os,requests,math,string,itertools,fractions,heapq,collections,re,array,bisect
from sqlalchemy import create_engine
from dateutil.relativedelta import relativedelta
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline
pd.set_option('display.max_columns', 500)
import matplotlib.dates as mdates

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
seed = 11389

df = pd.read_csv('../input/dataset_X.csv')

df.head()
df.describe()
dft = pd.DataFrame(StandardScaler().fit_transform(df.drop(['id'], axis = 1)))
dft.describe()
db = DBSCAN(eps=0.3, min_samples=100).fit(dft.sample(100000, random_state=0))
collections.Counter(db.labels_)