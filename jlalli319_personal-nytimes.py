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
from textblob import TextBlob



import warnings 

warnings.filterwarnings('ignore')



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

plt.style.use('fivethirtyeight')

%matplotlib inline
April_2017 = pd.read_csv('../input/ArticlesApril2017.csv')

print(April_2017.info())
April_2017.head()
April_2017.tail()
sorted(April_2017)
April_2017.shape
newDesk_df = April_2017[['newDesk']].copy()

newDesk_df.shape
newDesk_df.dtypes
newDesk_df['newDesk'].value_counts().plot(kind='bar')