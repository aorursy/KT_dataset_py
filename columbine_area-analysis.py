# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import altair as alt



import os, warnings

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

alt.renderers.enable('kaggle')
Referendum_Dataset = pd.read_csv('/kaggle/input/brexit-results/referendum.csv')



Referendum_Dataset.shape



Referendum_Dataset.columns
Referendum_Dataset.head(3)
Region_mean_rate = Referendum_Dataset.groupby('Region').mean()



Region_mean_rate['Region'] = Region_mean_rate.index



Region_mean_rate = Region_mean_rate[['Region', 'Percent Remain', 'Percent Leave']]
bar = alt.Chart(Region_mean_rate).mark_bar(

    color='lightblue'

).encode(

    x='Region',

    y='Percent Remain'

)



mean = alt.Chart(Region_mean_rate).mark_rule(

    color='black'

).encode(

    y='mean(Percent Remain)'

)

(bar+mean).properties(width=400, height=300)
bar = alt.Chart(Region_mean_rate).mark_bar(

    color='lightblue'

).encode(

    x='Region',

    y='Percent Leave'

)



mean = alt.Chart(Region_mean_rate).mark_rule(

    color='black'

).encode(

    y='mean(Percent Leave)'

)

(bar+mean).properties(width=400, height=300)
Referendum_Dataset.head(3)
Statistic = Referendum_Dataset[['Region', 'Remain', 'Leave', 'Rejected Ballots']]



Statistic.head(4)



attr_list = []



for i in Statistic.index:

    Region, Remain, Leave, Rejected = Statistic.loc[i]

    attr_list.append([Region, 'Remain', Remain])

    attr_list.append([Region, 'Leave', Leave])

#     attr_list.append([Region, 'Rejected Ballots', Rejected])

    

Statistic = pd.DataFrame(attr_list, columns=['Region', 'Type', 'People'])
bars = alt.Chart(Statistic).mark_bar().encode(

    x=alt.X('sum(People)', stack='zero'),

    y=alt.Y('Region'),

    color=alt.Color('Type')

)



text = alt.Chart(Statistic).mark_text(dx=-15, dy=3, color='white').encode(

    x=alt.X('sum(People):Q', stack='zero'),

    y=alt.Y('Region'),

    detail='Type',

    text=alt.Text('sum(People):Q', format='i')

)



(bars + text).properties(width=800, height=300)