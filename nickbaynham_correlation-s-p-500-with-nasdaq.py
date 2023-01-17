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
%matplotlib inline

import pandas as pd 

path = '/kaggle/input/correlation/'

def get_returns(file): 

    """

    This function get_data reads a data file from disk 

    and returns percentage returns.

    """    

    return pd.read_csv(path + file + '.csv', index_col=0, parse_dates=True).pct_change()





# Get the S&P time series from disk

df = get_returns('sp500') 



# Add a column for the Nasdaq

df['NDX'] = get_returns('ndx') 



# Calculate correlations, plot the last 200 data points.

df['SP500'].rolling(50).corr(df['NDX'])[-200:].plot()
