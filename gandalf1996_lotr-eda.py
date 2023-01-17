!pip install chart-studio
# Data processing libraries

import numpy as np 

import pandas as pd 



# Visualization libraries

import datetime

import matplotlib

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

sns.set()



# Plotly visualization libraries

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import chart_studio.plotly as py

from plotly.graph_objs import *

from IPython.display import Image

pd.set_option('display.max_rows', None)



import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
char_df = pd.read_csv('../input/lord-of-the-rings-data/lotr_characters.csv')

script_df = pd.read_csv('../input/lord-of-the-rings-data/lotr_scripts.csv')
char_df.head()
script_df.head()
records = script_df.groupby(['movie']).size()

records = records.sort_values()



grouped_df = pd.DataFrame(records)



grouped_df['Count'] = pd.Series(records).values

grouped_df['Movies'] = grouped_df.index

grouped_df['Log Count'] = np.log(grouped_df['Count'])

grouped_df.head()
fig = go.Figure(go.Bar(

    x = grouped_df['Movies'],

    y = grouped_df['Log Count'],

    text=['Bar Chart'],

    name='LOTR Movies',

    marker_color=grouped_df['Count']

))



fig.update_layout(

    height=800,

    title_text='Movies distribution in the LOTR Trilogy',

    showlegend=True

)



fig.show()
char_df.head()
gender_df = char_df[['gender','name', 'spouse']]

gender_df.head()
gen_df = gender_df.groupby('gender')['name'].value_counts().reset_index(name='count')

gen_df['count'] = gender_df.groupby('gender')['name'].transform('size')

gen_df.head()
test_df = gender_df

df = test_df.groupby(['gender'], as_index=False, sort=False)['name'].count()

df.head()
fig = px.pie(df, values='name', names='gender')

fig.show()
tdf = char_df.groupby(['race'], as_index=False, sort=False)['name'].count()

tdf.head()
fig = px.pie(tdf, values='name', names='race')

fig.show()
char_df.head()
script_df.head()
sdf = script_df.groupby('char')['movie'].value_counts().reset_index(name='count')

sdf['count'] = script_df.groupby('char')['movie'].transform('size')



sdf.head()
fig = px.pie(sdf, values='count', names='char')

fig.show()