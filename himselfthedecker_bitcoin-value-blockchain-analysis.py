import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn import preprocessing

pd.set_option('display.float_format', lambda x: '%.4f' % x)
dfBlockchainData = pd.read_csv('../input/bitcoin_dataset.csv')
dfBlockchainData.set_index('Date', inplace=True)
nRow, nCol = dfBlockchainData.shape
minFilled = dfBlockchainData.describe()[(dfBlockchainData.describe().index == 'count')].T.min()['count']
print(f'Dataset contains {nRow} lines (at least {((minFilled/nRow) * 100).round(4)}% completed) and {nCol} columns, described below')
print(dfBlockchainData.info())