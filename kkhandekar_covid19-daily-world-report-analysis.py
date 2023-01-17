import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualisation Libraries

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib import cm

import seaborn as sns

import warnings

import re



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-darkgrid')

pd.set_option('display.max_columns', 50)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.0f}'.format
url = '../input/coronavirus-source-data-covid19-daily-reports/full_data.csv'

data = pd.read_csv(url, header='infer')
data.shape
#Checking for null/ missing values

data.isnull().sum()
data = data.fillna(0)
cols = set(data.columns)

col_numeric = set(['new_cases','new_deaths','total_cases','total_deaths'])



for x in col_numeric:

    data[x] = data[x].astype('int')

data.dtypes
data.head()
#Creating a function to visualize Location wise



def visualise(loc):

    

    loc_df = data[data['location'] == loc]   #creating a seperate dataframe

    loc_df.index = pd.DatetimeIndex(loc_df['date'])  #converting the date column to index

    loc_df.drop(['date'], axis=1, inplace=True)      # dropping the original date columns

    

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))

    

    ax.plot(loc_df['total_cases'].resample('D').sum()) 

    ax.plot(loc_df['total_deaths'].resample('D').sum(), color='red') 

    

    ax.set_ylabel('Cases')

    ax.set_title(f'{loc.capitalize()} - Total Cases vs Total Deaths', fontsize=16)

    ax.legend(prop={'size':12},bbox_to_anchor=(1, 0, 0.15, 1),shadow=True)

    ax.tick_params(labelsize=10, rotation=45)

    

    fig.tight_layout()

    plt.show()

#visualising USA

visualise('United States')
#visualising Brazil

visualise('Brazil')
#Visualising Nigeria

visualise('Nigeria')
#China

visualise('China')
#Italy

visualise('Italy')
#Iran

visualise('Iran')
#Australia

visualise('Australia')
#India

visualise('India')