# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install bar_chart_race
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import bar_chart_race as bcr

from IPython.display import HTML

import warnings

warnings.filterwarnings("ignore")

from IPython.display import Video
covid_df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv', index_col=False)

covid_df.head()
covid_df.info()
# change datatype of date to a pandas datetime format

covid_df['Date'] = pd.to_datetime(covid_df['Date'], dayfirst=True)



#covid_data_complete["Date"] = covid_data_complete["Date"].apply(pd.to_datetime)



#drop columns other than total_cases

drop_cols = ['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational', 'Cured', 'Deaths']

covid_df.drop(covid_df[drop_cols],axis=1,inplace=True)





covid_df = covid_df[covid_df['State/UnionTerritory'] != 'Cases being reassigned to states']



#dropping data before Feb 29 2020 as only Kerala had 3 cases till then in a span of one month

covid_df = covid_df[covid_df['Date'] > pd.to_datetime(pd.DataFrame({'year': [2020],'month': [2],'day': [28]}))[0]]



covid_df.head()
covid_data = covid_df.copy() #make a copy for analysis

covid_data.columns = ['Date', 'States', 'Cases'] #rename columns

covid_data.head(10)
total_states = covid_data['States'].nunique()

total_states
# set states and date as index and find cases

# transpose the dataframe to have countries as columns and dates as rows

covid_data_by_date = covid_data.set_index(['States','Date']).unstack()['Cases'].T.reset_index()



covid_data_by_date = covid_data_by_date.set_index('Date') #make date as index - desired by barchartrace



covid_data_by_date = covid_data_by_date.fillna(0) #fill na with 0



covid_data_by_date
#make the mp4 file with the BarChartRace and save it



bcr.bar_chart_race(

    df=covid_data_by_date,

    filename='India_Covid19_BarChartRace.mp4',

    orientation='h',

    sort='desc',

    n_bars=10,

    fixed_order=False,

    fixed_max=False,

    steps_per_period=10,

    interpolate_period=False,

    label_bars=True,

    bar_size=.95,

    period_label={'x': .99, 'y': .25, 'ha': 'right', 'va': 'center'},

    period_fmt='%B %d, %Y',

    period_summary_func=lambda v, r: {'x': .99, 'y': .05,

                                      's': f'Total cases: {v.nlargest(total_states).sum():,.0f}',

                                      #'s': '',

                                      'ha': 'right', 'size': 10, 'family': 'Courier New'},

    perpendicular_bar_func='median',

    period_length=1000,

    figsize=(5, 3),

    dpi=500,

    cmap='dark24',

    title='COVID-19 cases in India',

    title_size=10,

    bar_label_size=10,

    tick_label_size=10,

    shared_fontdict={'color' : '.1'},

    scale='linear',

    writer=None,

    fig=None,

    bar_kwargs={'alpha': .7},

    filter_column_colors=True) 