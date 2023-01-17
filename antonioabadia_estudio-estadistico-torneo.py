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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf
%matplotlib inline
df=pd.read_csv('../input/torneo/results.csv', delimiter=',')
print('Number of rows and columns:', df.shape)
df.head(5)
print(df.columns.tolist())
df.describe()
df.info()
sns.distplot(df['action'], kde=False);
plt.figure(figsize=(30,5))
sns.boxplot(x="id", y="action", data=df)
plt.xticks(rotation=90)
plt.xlabel('id')
plt.title('Box plot action/id')
sns.despine(left=True)
plt.tight_layout()
plt.figure(figsize=(30,10))
sns.violinplot(x="id", y="action", data=df)
plt.xticks(rotation=90)
plt.xlabel('id')
plt.title('Box plot action/id')
sns.despine(left=True)
plt.tight_layout()
plt.figure(figsize=(30,5))
sns.boxplot(x="round", y="action", data=df)
plt.xlabel('roud')
plt.title('Box plot action/round')
sns.despine(left=True)
plt.tight_layout()

plt.figure(figsize=(30,5))
sns.violinplot(x="round", y="action", data=df)
plt.xlabel('roud')
plt.title('violin plot action/round')
sns.despine(left=True)
plt.tight_layout()

# plt.figure(figsize=(14,5))
# sns.relplot(x="game", y="action", estimator=None, kind="line", data=df);
# sns.despine(left=True)
# plt.tight_layout()

plt.figure(figsize=(30,5))
sns.relplot(x="game", y="action", kind="line", ci=None, data=df);
sns.despine(left=True)
plt.tight_layout()