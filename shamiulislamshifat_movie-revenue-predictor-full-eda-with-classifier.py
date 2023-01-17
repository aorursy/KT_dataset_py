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
#data processing libraries
import json
import datetime
import ast
import pandas as pd
import numpy as np
from scipy import stats
from wordcloud import WordCloud, STOPWORDS

#plotting libraries
%matplotlib inline
from IPython.display import Image, HTML
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
!pip install chart_studio
import chart_studio
chart_studio.tools.set_credentials_file(username='shamiulshifat', api_key='mEJeIwveMaaGUP7c86Qe')
sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)

#ML libraries
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
df_metadata=pd.read_csv('../input/the-movies-dataset/movies_metadata.csv')
df_metadata.head(5)
df_metadata.columns
df_metadata.info()
#lets remove unnecessary columns which are not needed!
df_metadata=df_metadata.drop(['imdb_id'], axis=1)
#lets drop oriiginal_title" column as we only prefer english title
df_metadata=df_metadata.drop(['original_title'], axis=1)
df_metadata=df_metadata.drop(['adult'], axis=1)
df_metadata[df_metadata['revenue']!=0].shape
df_metadata['revenue']=df_metadata['revenue'].replace(0, np.nan)
df_metadata['budget']=pd.to_numeric(df_metadata['budget'], errors='coerce')
#now replace to NaN values
df_metadata['budget']=df_metadata['budget'].replace(0, np.nan)
#now check null values
df_metadata[df_metadata['budget'].isnull()].shape
#return can be calculated by revenue/budget
df_metadata['return']=df_metadata['revenue']/df_metadata['budget']

#lets look into release date
#transform date time into more pythonic way

df_metadata['release_date']=pd.to_datetime(df_metadata['release_date'], errors='coerce').apply(lambda x:str(x).split('-')[0] if x!=np.nan else np.nan)
#lets convert to string type
df_metadata['title']=df_metadata['title'].astype('str')
df_metadata['overview']=df_metadata['overview'].astype('str')

#lets join titles continuousy seperated by space
title_data=' '.join(df_metadata['title'])
overview_data=' '.join(df_metadata['overview'])


title_cloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=1000, width=3000).generate(title_data)
plt.figure(figsize=(12,6))
plt.imshow(title_cloud)
plt.axis('off')
plt.show()
overview_cloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=1000, width=3000).generate(overview_data)
plt.figure(figsize=(12,6))
plt.imshow(overview_cloud)
plt.axis('off')
plt.show()
#we will build a seperate dataframe-countries_data
df_metadata['production_countries']=df_metadata['production_countries'].fillna('[]').apply(ast.literal_eval)

df_metadata['production_countries'] = df_metadata['production_countries'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
countries = df_metadata.apply(lambda x: pd.Series(x['production_countries']),axis=1).stack().reset_index(level=1, drop=True)
countries.name = 'countries'
#lets build data to plot
#we will build custom dataframe

countries_data = df_metadata.drop('production_countries', axis=1).join(countries)
countries_data = pd.DataFrame(countries_data['countries'].value_counts())
#lets build a column with index
countries_data['country'] = countries_data.index
countries_data.columns = ['num_movies', 'country']
countries_data = countries_data.reset_index().drop('index', axis=1)
countries_data.head(10)

#lets plot using geomap
#you can choose any country.. we are excluding 2nd most popular country
countries_data = countries_data[countries_data['country'] != 'United Kingdom']

#lets build a dict with parameters for geoplot
country_data = [ dict(
        type = 'choropleth',
        locations = countries_data['country'],
        locationmode = 'country names',
        z = countries_data['num_movies'],
        text = countries_data['country'],
        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(255, 0, 0)']],
        autocolorscale = False,
        reversescale = False,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Production Countries'),
      ) ]
#layout design

layout = dict(
    title = 'Production Countries for the MovieLens Movies (Excluding UK)',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=country_data, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )
