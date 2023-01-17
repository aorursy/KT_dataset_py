# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# os.rename('Report-66-2704201512525131PM-2012-2013.csv','data.csv')
# Any results you write to the current directory are saved as output.
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import models
from keras import layers
import keras
from __future__ import print_function
import statsmodels.api as sm
import pandas
from patsy import dmatrices
import seaborn as sns
from io import StringIO
import datetime
import re
import json
import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout,Embedding,Flatten
import warnings
from pprint import pprint
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#Caution.
# import warnings
# warnings.filterwarnings("ignore")

# mpl.rcdefaults()

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
# keras.__version__

# import time
# tic = time.time()
# a = [(4+i)**3 for i in range(10000)]
# toc = time.time()
# (toc - tic)*1000 
# %timeit a = [(4+i)**3 for i in range(10000)]

#set default behaviour
#mpl.rcdefaults()
df = pd.read_csv('../input/Report-66-2704201512525131PM-2012-2013.csv',encoding = 'unicode_escape')
df.head(5)
df['UId'] = df['University Name'].map(lambda x : x[-7:-1])
df['CId'] = df['College Name'].map(lambda x : x[-8:-1])
# df['CId']
df.shape
df.columns
del df['S. No.']
#No of states
len(df['State Name'].unique())
#actual - 36(states + union territories)
#No of districts
len(df['District Name'].unique())
#actual 640 districts
#no of colleges
len(df['College Name'].unique())
#no of universities
len(df['University Name'].unique())
import matplotlib as mpl
mpl.rcdefaults()
#Statewise distribution of colleges
df['State Name'].value_counts().plot(kind = 'bar')
plt.show();
df['College Type'].value_counts().plot(kind = 'bar')
plt.show();
#most of the colleges are affiliated
#How many university in each state
df.groupby(['State Name'])['University Name'].nunique().sort_values().plot(kind = 'bar')
plt.show();
#university in Uttar Pradesh
len(df[df['State Name']=='Uttar Pradesh']['University Name'].unique())
#which university has highest affiliated colleges. since 338 colleges are so for clarity we're taking only highest 10 and lowest 10
df.groupby(['University Name'])['College Name'].count().sort_values(ascending = False)[:10].plot(kind = 'bar')
plt.show();
df.groupby(['University Name'])['College Name'].count().sort_values(ascending = False)[-10:].plot(kind = 'bar')
#they don't have any affiliated college.
plt.show();
#which university is present in highest no of districts
df.groupby(['University Name'])['District Name'].nunique().sort_values()[-10:].plot(kind= 'bar')
plt.show();
len(df[df['UId'] == 'U-0283']['District Name'].unique()) #verified
#to be continued
