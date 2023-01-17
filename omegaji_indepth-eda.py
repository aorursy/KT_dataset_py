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
import pandas as pd

import numpy as np

from scipy import stats





import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected = True) 



from IPython.display import Markdown

def bold(string):

    display(Markdown(string))

df_train = pd.read_csv('../input/bigquery-geotab-intersection-congestion/train.csv')

df_test = pd.read_csv('../input/bigquery-geotab-intersection-congestion/test.csv')

def description(df):

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    

    summary['First Value'] = df.iloc[0].values

    summary['Fifth Value'] = df.iloc[5].values

    summary['Third Value'] = df.iloc[2].values

    return summary
bold(' Description of  train Data:')

display(description(df_train))

bold(' Description of  test Data:')

display(description(df_test))

print(df_train.shape)
df_train.dropna(inplace=True)

print(df_train.shape)

df_train["EntryStreetName"]

df_train["freq"]=df_train.groupby(["EntryStreetName"])["EntryStreetName"].transform('count')

df_train=df_train.sort_values(by=["freq"])

fig,ax=plt.subplots(2,1,figsize=[13,12])

sns.set_style("white")

plt.subplots_adjust(hspace=0.8)

df_train["EntryStreetName"].value_counts()[:70].plot(kind="bar",ax=ax[0],fontsize=8)

df_train["ExitStreetName"].value_counts()[:70].plot(kind="bar",ax=ax[1],fontsize=8)

sns.set_style("dark")

fig, ax = plt.subplots(2,1, figsize=[15, 12])



sns.countplot(data = df_train, x = 'EntryHeading', ax = ax[0], palette = 'YlOrRd_r')

ax[0].set_title('Count plot of Entry Heading', fontsize = 22)

ax[0].set_xlabel('Entry Heady', fontsize = 18)



sns.countplot(data = df_train, x = 'ExitHeading', ax = ax[1], palette = 'YlGnBu')

ax[1].set_title('Count plot of Exit Heading', fontsize = 22)

ax[1].set_xlabel('Exit Heady', fontsize = 18)



plt.subplots_adjust(hspace = 0.3)

plt.show()
df_train.head()