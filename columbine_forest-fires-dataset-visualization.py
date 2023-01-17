# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import altair as alt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os,warnings

warnings.filterwarnings('ignore')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



alt.renderers.enable('kaggle')
# we should change the encoding format when we using read_csv api.

forest_fires_dataset = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv", encoding='latin')



print(forest_fires_dataset.shape)

print(forest_fires_dataset.columns)

forest_fires_dataset.head(10)
# we first select the needed column for our task.

state_count = forest_fires_dataset[['state', 'number']]



state_count = state_count.groupby('state').sum()



state_count['state'] = state_count.index



state_count.index = range(state_count.shape[0])



state_count.head(4)
import altair as alt



bar_graph = alt.Chart(state_count).mark_bar(

    color='lightblue'

).encode(

    x='number',

    y='state',

).properties(width=600)



mean_line = alt.Chart(state_count).mark_rule(

    color='black'

).encode(

    x = 'mean(number)'

)



annotation = bar_graph.mark_text(

    align='left',

    baseline='middle',

    dx=3

).encode(

    text='number'

)



bar_graph + mean_line + annotation
Mato_Data = forest_fires_dataset.loc[forest_fires_dataset['state']=='Mato Grosso']
Mato_Data_year = Mato_Data.groupby('year').sum()



Mato_Data_year['year'] = Mato_Data_year.index



Mato_Data_year.index = range(Mato_Data_year.shape[0])



Mato_Data_year = Mato_Data_year.sort_values('year')



Mato_Data_year.head(5)
line_graph = alt.Chart(Mato_Data_year).mark_area(

    color="lightblue",

    interpolate='step-after',

    line=True

).encode(

    x='year',

    y='number'

)

line_graph
city_year_Data = forest_fires_dataset.groupby(['state', 'year']).sum()



city_year_Data['state'] = [city_year_Data.index[i][0] for i in range(city_year_Data.shape[0])]



city_year_Data['year'] = [city_year_Data.index[i][1] for i in range(city_year_Data.shape[0])]



city_year_Data.index = range(city_year_Data.shape[0])

line_graph = alt.Chart(city_year_Data).mark_line().encode(

    x='year',

    y='number',

    color='state'

)

line_graph.properties(width=800, height=400)
line_graph = alt.Chart(city_year_Data.loc[city_year_Data['state']=='Acre']).mark_line().encode(

    x='year',

    y='number',

    color='state'

)

line_graph
Mato_Data.head(3)

Mato_Data_month = Mato_Data.groupby('month').sum()



Mato_Data_month['month'] = Mato_Data_month.index



Mato_Data_month.index = range(Mato_Data_month.shape[0])



Mato_Data_month.head(3)
%pylab inline



figure = plt.figure(figsize=(10,10))

plt.title('Mato Grosso Forest Fires in each Month')

plt.pie(Mato_Data_month['number'], labels=Mato_Data_month['month'])

plt.show()