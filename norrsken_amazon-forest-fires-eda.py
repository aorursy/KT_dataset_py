# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from ipywidgets import interact

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# importing the dataset.

df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

df = df.drop('date', axis = 1) #removing 'date' column from dataset as it contains the date 1st Jan of fire year.

months = {'Janeiro':'Jan', 'Fevereiro':'Feb', 'Março':'Mar', 'Abril':'Apr', 'Maio':'May', 'Junho':'June', 'Julho': 'July',

       'Agosto':'Aug', 'Setembro':'Sept', 'Outubro':'Oct', 'Novembro': 'Nov', 'Dezembro':'Dec'}

df['month'] = df['month'].apply(lambda x: months[x]) #translating month name to English.

df['number'] = df['number'].round(0) # Rounding off the number of forest fires to integers

df.head()
# Plotting yearly count trend of fires.

fires = df[['year', 'number']].groupby(['year']).sum()

fires = fires.reset_index()

state_wise = df.groupby(['state', 'year']).sum().reset_index()
# transforming the data into a matrix with columns representing years and rows containing states' fire counts for corresponding year.

state_wise_yearly = state_wise.pivot(index = 'state', columns = 'year', values = 'number')

state_wise_yearly



# code to highlight maximum fires count each year.

def highlight(s):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    return ['background-color: orange' if v else '' for v in is_max]

state_wise_yearly.style.apply(highlight)
state = state_wise[['state', 'number']].groupby(['state']).sum().sort_values(by = ['number'], ascending = False).head(10).reset_index()

plt.figure(figsize = (16,6))

ax = sns.barplot(x = 'state', y = 'number', data = state, palette = 'coolwarm_r')

ax.set(title = 'Statewise count of forest fires (1998 - 2017)', xlabel = 'State', ylabel = 'Forest Fires')
#storing longitudinal and latitudinal coordinates for the ten states

state['lat'] = [-16.350000, -22.15847, -23.533773, -22.908333, -11.409874, -21.5089, -16.328547, -19.841644, -21.175, -3.416843]

state['lon'] = [-56.666668, -43.29321, -46.625290, -43.196388, -41.280857, -43.3228, -48.953403, -43.986511, -43.01778, -65.856064]

state["number"] = pd.to_numeric(state["number"]) #casting column values to numeric type



fig = px.scatter_geo(state,  # Input Pandas DataFrame

                    color="state",

                    lat = 'lat',

                    lon = 'lon',

                     size = 'number',# DataFrame column with color values

                    hover_name="state") # displays state name and other details on hovering over the map.

fig.update_layout(

    title_text = 'Number Of Fires', # Create a Title

    geo_scope='south america',  # Plot only South America instead of globe

)

fig.show()
# interactive graph showing yearly fire count of each state relative to total number of fires.

@interact (state = ['Acre', 'Alagoas', 'Amapa', 'Amazonas', 'Bahia', 'Ceara',

       'Distrito Federal', 'Espirito Santo', 'Goias', 'Maranhao',

       'Mato Grosso', 'Minas Gerais', 'Paraiba', 'Pará', 'Pernambuco',

       'Piau', 'Rio', 'Rondonia', 'Roraima', 'Santa Catarina',

       'Sao Paulo', 'Sergipe', 'Tocantins'])

def plot_graphs(state):

    plt.figure(figsize = (16, 6))

    sns.set_style("whitegrid")

    ax = sns.lineplot(x='year', y='number', data = state_wise[state_wise['state'] == state], color = 'orange', lw = 4, hue = 'state')

    ax = sns.lineplot(x='year', y='number', data = fires, color = 'red', lw = 4, label = 'Total Fires')

    ax.set(xlabel='Year', ylabel='Number of Fires', title = 'Forest Fires in ' + state)

    ax.xaxis.set_major_locator(plt.MaxNLocator(21))

    ax.yaxis.set_major_locator(plt.MaxNLocator(15))

    ax.set_xlim(1998, 2018)

    ax.legend()

    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    plt.show()
# monthly average number of fires during period 1998 - 2017

month_wise = df[['month', 'number']].groupby(['month']).mean().reset_index()

plt.figure(figsize = (16,10))

ax = sns.barplot(x = 'month', y = 'number', data = month_wise, palette = 'magma_r', order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])

ax.set(title = 'Average number of forest fires by month', xlabel = 'Number of fires', ylabel = 'Month')