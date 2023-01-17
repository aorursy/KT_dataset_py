# Install package in the current Jupyter kernel

import sys

!{sys.executable} -m pip install bar_chart_race
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import bar_chart_race as bcr   # use pip install bar_chart_race in console to install this package

from IPython.display import HTML

import warnings

warnings.filterwarnings("ignore")

from IPython.display import Video
covid_data_complete = pd.read_csv('../input/covid19-cases-by-country/covid-data.csv')

covid_data_complete.head()
covid_data_complete.info()
covid_data_complete.isnull().sum()
# change datatype of date to a pandas datetime format

covid_data_complete["date"] = covid_data_complete["date"].apply(pd.to_datetime, dayfirst=True)



#drop columns other than total_cases

drop_cols = ['new_cases', 'total_deaths', 'new_deaths']

covid_data_complete.drop(covid_data_complete[drop_cols],axis=1,inplace=True)

#covid_data_complete.drop(['new_cases' , 'total_deaths', 'new_deaths'] , axis=1, inplace=True)



covid_data_complete.head()
covid_data = covid_data_complete.copy() #make a copy for analysis

covid_data.columns = ['Date', 'Countries', 'Cases'] #rename columns

covid_data.head(10)
total_countries = covid_data['Countries'].nunique()

total_countries
covid_data.set_index(['Countries','Date']).unstack()['Cases'].T.reset_index()
# set countries and date as index and find cases

# transpose the dataframe to have countries as columns and dates as rows

covid_data_by_date = covid_data.set_index(['Countries','Date']).unstack()['Cases'].T.reset_index()



covid_data_by_date = covid_data_by_date.set_index('Date') #make date as index - desired by barchartrace



covid_data_by_date = covid_data_by_date.fillna(0) #fill na with 0

covid_data_by_date
#make the mp4 file with the BarChartRace and save it

df = covid_data_by_date

bcr.bar_chart_race(

    df=df,

    filename='Covid19_BarChartRace.mp4',

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

    period_summary_func=lambda v, r: {'x': .99, 'y': .18,

                                      's': '',

                                      'ha': 'right', 'size': 8, 'family': 'Courier New'},

    perpendicular_bar_func='median',

    period_length=1000,

    figsize=(5, 3),

    dpi=500,

    cmap='dark24',

    title='Developer: Akhil James\n  COVID-19 Cases by Country\n ',

    title_size=10,

    bar_label_size=7,

    tick_label_size=5,

    shared_fontdict={'color' : '.1'},

    scale='linear',

    writer=None,

    fig=None,

    bar_kwargs={'alpha': .7},

    filter_column_colors=True)
# from IPython.display import HTML

# HTML('<iframe width="900" height="600" src="https://www.kaggleusercontent.com/kf/35204744/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MesAbveuEYdnCB0xQCLoCA.gkUggoSk3_vIAjuQZr6VODLqU9DrDvj-Wk6_sQo3-ZUe3jfiODpEGcJwOlfnTbxECN2oeFKtiWXWfm3CdiRZzZO2P-JekAJt_7o0UG88HVDcK_xKBv_V5uVXZcd1RDqd7Csdeu03zEcBmy_li8b_M7L-Cs0bJz69IaWkn22lVlnikd-3MO9WXA0w5BVLVnNrTjnCLQoCQwD-xQ86YgaiYqCJoq9cMwCr4Gdr_n5r35qIHo2B8lCfdHwYCaSkAycMrd1rRdBmGwPuQ5gx_X6D7XCMcyac1KB99CCK266TfekBvDTFEUN3Qav7xclITcspN2KsVwHvfgezucOjmp1NiQ97aWIVKAs0NHHAB2uysuVb5zh9CXB1Wy7LyWWOTsQ449HXrTzcuCqjF8sHdzzrSdt13YR78rlGeU-aZKzqapGxc9LDk0PGoGXuHPYJEEtoisc_ZOTZha0eNTh8pSXU7VxT1lkF6qhgwQTnLKRNtPLM8o8IGtceu2WddpS27fMT8dXiuHk04lw3FTnq-58p2EI44qEKvTOTLFwCACGm7NDW1917Jku1y1doxbnZROhIKbmlvagKSXwsLRsOW_UHNqgFYig2JbSxq_pa0diGVce3voeMs5W9wkv6VvXHrNKViqibVstAwQ2MyDk3n5ofxm9x8rPMdoFIFcY6_IlL8sxUeVSr7SAv-THurkZoLOVo.yi6gIq3LNiCzn3s1MUeAbw/Covid19_BarChartRace.mp4" frameborder="0" allowfullscreen></iframe>')
#other methods to load video



# Video("../output/kaggle/working/Covid19_BarChartRace.mp4", width=900, height=600, embed=True)



#########################################################################################################



# %%HTML

# <video width=400 controls>

#   <source src="../output/kaggle/working/Covid19_BarChartRace.mp4" type="video/mp4">

# </video>



##########################################################################################################



# from IPython.display import HTML



# HTML("""

# <div align="middle">

# <video width="100%" controls>

#       <source src="../output/kaggle/working/Covid19_BarChartRace.mp4" type="video/mp4">

# </video></div>""")



##########################################################################################################



# ! ln -sf "../output/kaggle/working/Covid19_BarChartRace.mp4" ./Covid19_BarChartRace.mp4

# Video("Covid19_BarChartRace.mp4", embed=True)