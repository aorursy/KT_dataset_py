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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from mlxtend.preprocessing import minmax_scaling
from matplotlib.pyplot import plot
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import geopandas
import geoplot
import io
df=pd.read_csv('../input/world-happiness-report-2019/world-happiness-report-2019.csv')
df.head()
df.info()
df.shape
df.isnull().sum()
df.loc[df['Country (region)']=='Poland']
df.rename(columns = {'Country (region)' : 'Country', 'SD of Ladder' : 'SD_of_Ladder', 'Positive affect' : 'Positive_affect', 'Negative affect' : 'Negative_affect','Social support' : 'Social_support','Log of GDP\nper capita' : 'Log_GDP_per_capita','Healthy life\nexpectancy' : 'Healthy_life_expectancy'}, inplace = True)
df.head()
df[pd.isnull(df.Corruption)]
df.head()
original=df.copy()
def highlight_max(s):    
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

df.style.apply(highlight_max, subset=['Ladder','SD_of_Ladder','Positive_affect','Negative_affect','Social_support','Freedom','Corruption','Generosity','Log_GDP_per_capita','Healthy_life_expectancy'])
df.corr()
y,ax = plt.subplots(figsize=(8, 7))
sns.heatmap(df.corr(),annot=True, linewidths=2.50, fmt= '.1f',ax=ax, cmap="Reds")
plt.xticks(rotation=50) 
ax.set_title("Correlation between indicators", fontsize=30, color ='lightcoral')


y.tight_layout()
plt.show()
world = '../input/world-boundaries/countries_shp/countries.shp'
map_world = geopandas.read_file (world)

map_world.plot ()

map_world.head ()
map_world.CONTINENT.value_counts()
merged = df.set_index('Country').join(map_world.set_index('COUNTRY'))
merged.head()
merged.info()
data = dict(type = 'choropleth', 
           colorscale = 'RdBu',
           locations = merged['SOVEREIGN'],
           locationmode = 'country names',
           z = merged['Ladder'],
           colorbar = {'title':'Place in the ranking'})
layout = dict(title = 'Global Happiness Ranking',
             titlefont=dict(size=30),
             title_font_family="Times New Roman",
             title_font_color="lightcoral",
             geo = dict(showframe = True, 
                       projection = {'type': 'natural earth'}))
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)
data = dict(type = 'choropleth', 
           colorscale = 'turbid',
           locations = merged['SOVEREIGN'],
           locationmode = 'country names',
           z = merged['Corruption'],
           colorbar = {'title':'Corruption'})
layout = dict(title = 'Global Happiness Ranking - corruption', 
             titlefont=dict(size=30),
             title_font_family="Times New Roman",
             title_font_color="maroon",
             geo = dict(showframe = True, 
                       projection = {'type': 'natural earth'}))
choromap = go.Figure(data = [data], layout = layout)
iplot(choromap)