%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

plt.style.use('ggplot')



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import plotly.figure_factory as ff

import cufflinks as cf



import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('../input/data.csv')
df.head()
df.columns
df.describe()
df.dropna(how='all')
df.describe()
#EDA
df_top200 = df.sort_values(by=['Overall'], ascending=False).head(200)
df_top200['Nationality'].value_counts().plot.bar()
data = [go.Histogram(

        x = df_top50.Overall,

        xbins = {'start': 70, 'size': 1, 'end' :95}

)]

plotly.offline.iplot(data, filename='overall_rating_distribution')
number_of_apps_in_category = df_top200['Club'].value_counts().sort_values(ascending=False).head(15)

data = [go.Pie(

        labels = number_of_apps_in_category.index,

        values = number_of_apps_in_category.values,

        hoverinfo = 'label+value'

    

)]



plotly.offline.iplot(data, filename='active_category')

footballers = df_top200.copy()

footballers['Unit'] = df_top200['Value'].str[-1]

footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 

                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

footballers['Value (M)'] = footballers['Value (M)'].astype(float)

footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 

                                    footballers['Value (M)'], 

                                    footballers['Value (M)']/1000)
import seaborn as sns



sns.lmplot(x='Value (M)', y='Overall', hue='Position', 

           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], 

           fit_reg=False)
#Who's best at every position
df_sort = df.sort_values(by=['Overall','Potential'],ascending=False)
all_position = df.Position.unique()[:-1]



columns = ['Position','Name', 'Overall','Club']

df_ = pd.DataFrame(columns=columns)



for position in all_position:

    x = df_sort.loc[df_sort['Position'] == position][:1]

    df2 = pd.DataFrame(x[columns])

    df_ = df_.append(df2)

df_