# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import re

import os

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

import plotly.figure_factory as ff

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn.model_selection import StratifiedShuffleSplit

from xgboost import XGBRegressor



import matplotlib.pyplot as plt

%matplotlib inline

import scipy

import scipy.stats as st

from xgboost import plot_importance



from datetime import datetime

        

# This will help to have several prints in one cell in Jupyter.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



# don't truncate the pandas dataframe.

# so we can see all the columns

pd.set_option("display.max_columns", None)



import warnings

warnings.filterwarnings('ignore')
nfl_df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
nfl_df.head()
nfl_df.columns
nfl_df.shape
nfl_df.describe().transpose()
# This table gives you the complete picture 

# Howmany unique values for each feature, what is it ? and what is the frequency of it ? 

table_Cate = ff.create_table(nfl_df.describe(include=['O']).T, index = True, index_title = 'Categorical Columns')

iplot(table_Cate)
nfl_df['Location'].value_counts()
nfl_df['StadiumType'].value_counts()
nfl_df['PossessionTeam'].value_counts()
nfl_df['PossessionTeam'].unique()
nfl_df['HomeTeamAbbr'].unique()
maping_state = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

for abbr in nfl_df['PossessionTeam'].unique():

    maping_state[abbr] = abbr
nfl_df['PossessionTeam'] = nfl_df['PossessionTeam'].map(maping_state)
nfl_df['PossessionTeam'].unique()
nfl_df['HomeTeamAbbr'] = nfl_df['HomeTeamAbbr'].map(maping_state)

nfl_df['VisitorTeamAbbr'] = nfl_df['VisitorTeamAbbr'].map(maping_state)
nfl_df['StadiumType'].unique()
nfl_df['StadiumType'].value_counts()
#Coverting the values to Lower case for ease of further processing

nfl_df['StadiumType'] = nfl_df['StadiumType'].apply(lambda x : x.lower() if not pd.isna(x) else x)
nfl_df['StadiumType'] = nfl_df['StadiumType'].apply(lambda x : 'Outdoor' if (not pd.isna(x)) and ('out' in x 

                                                    or 'open' in x

                                                    or 'cloudy' in x

                                                    or 'hein' in x)

                                                    else 'Indoor'

                                                    

                                                    

                                                    

                                                     )
nfl_df['GameWeather'].value_counts()
from collections import Counter

weather_count = Counter()

for weather in nfl_df['GameWeather']:

    if pd.isna(weather):

        continue

    for word in weather.split():

        weather_count[word]+=1

weather_count.most_common()[:15]
nfl_df['GameWeather'].unique()
#Converting every word to lower alpha bets

nfl_df['GameWeather'] = nfl_df['GameWeather'].str.lower()
