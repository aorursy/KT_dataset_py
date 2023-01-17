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
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
df = pd.read_pickle('../input/all_banks')
df.head()
tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']
df.xs(key='Close',axis=1,level='Stock Info').max()
returns = pd.DataFrame()
for tick in tickers:
        returns[tick+' Return'] = df[tick]['Close'].pct_change()
returns.head()
sns.pairplot(returns[1:])
returns.idxmin()
returns.idxmax()
returns.std()
returns.ix['2015-01-01':'2015-12-31'].std()
sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)
sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()
for tick in tickers:
    df[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()
sns.heatmap(df.xs(key='Close',axis=1,level='Stock Info').corr(),cmap='coolwarm',annot=True)
sns.clustermap(df.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)
