# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



!pip install plotly==4.4.1

!pip install chart_studio

!pip install xlrd
import os

import tk_library_py

from tk_library_py import excel_to_df
for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_excel("/kaggle/input/treasury/Datathon_Treasury interest rates-.xlsx")

data

%matplotlib inline

data[["Bid Yield"]].plot()
from pandas import Series, DataFrame

 

import matplotlib.pyplot as plt
from pandas_datareader import data, wb

from datetime import datetime

 

end = datetime.now()

start = datetime(end.year - 20, end.month, end.day)

SP = data.DataReader('^GSPC', 'yahoo', start, end)
SP['Adj Close'].plot(legend=True, figsize=(10,4))

plt.show() 