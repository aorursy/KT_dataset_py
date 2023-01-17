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
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv')
df.head()
df.info()
df.shape
df['month'] = pd.to_datetime(df['date']).dt.month
df['year'] = pd.to_datetime(df['date']).dt.year
msno.matrix(df)
missing_percentage=df.isna().sum()*100/df.shape[0]
missing_percentage
df.dropna(inplace=True)
cardinality = {}
for col in df.columns:
    cardinality[col] = df[col].nunique()

cardinality
df.shape
df['race'].unique()
killing_by_race=df[df['manner_of_death']=='shot']
fig = px.histogram(killing_by_race,x='race',color='race')
fig.show()
df.manner_of_death.count()
shootout_by_states = df['state'].value_counts()[:10]
shootout_by_states = pd.DataFrame(shootout_by_states)
shootout_by_states=shootout_by_states.reset_index()
fig = px.pie(shootout_by_states, values='state', names='index', color_discrete_sequence=px.colors.sequential.RdBu)
fig.show()