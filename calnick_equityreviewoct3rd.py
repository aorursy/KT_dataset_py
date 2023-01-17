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
# <1> 

# Import libraries

import os

import pandas as pd



import numpy as np





from pandas import ExcelWriter

from pandas import ExcelFile



# matplot libs

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt



# additional Data viz libraries



import seaborn as sns

cwd = os.getcwd()

cwd
os.listdir(cwd)
# load equity prices

EQ = pd.read_excel(r"/kaggle/input/Equity_Oct3rd.xlsx")
EQ.head()
# Make the EQ df column names unique

EQ.columns = [str(col) + '_EQ' for col in EQ.columns]



# Rename Date column

EQ.rename(columns={'Date_EQ':'Date'}, inplace=True)
EQ.head()
#All.head()

type(EQ)
# define a functoin named "calculate percentage"



def calculate_percentage(price_list):

    pct_change = []

    prev_day_price = np.NaN

    for price in price_list:

        # declare previous year tuition as a global variable, not just a local one within this function

        #global prev_day_price

        pct_change.append((price - prev_day_price) *100 / prev_day_price)

        prev_day_price = price

    # now that the percent list has been created, round these variables 

    pct_change = [round(x,2) for x in pct_change]

    return pct_change
# set date as in the index column

#df = All

df=pd.DataFrame(EQ).set_index("Date")
# iterating the columns 

for col in df.columns: 

    print(col) 
# Let's see how all thee banks correlation matrix

df.corr()
# show a correlation plot between all the columns

corr = df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
import pandas as pd

import statsmodels.api as sm

#from statsmodels.tsa.seasonal import STL

from sklearn.metrics import mean_squared_error

from statsmodels.tsa.ar_model import AR



import matplotlib.pyplot as plt

import seaborn as sns



# include below line if you are using Jupyter Notebook

#%matplotlib inline



# Set figure width to 12 and height to 9

plt.rcParams['figure.figsize'] = [12, 9]



df = pd.read_csv('EURUSD.csv',sep='\t', index_col='Date')

df.index = pd.to_datetime(df.index)

df.sort_index(inplace=True)

df = df.resample('W').last()

series = df['Price']