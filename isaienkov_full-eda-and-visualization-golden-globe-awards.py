# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import lightgbm as lgb

from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.figure_factory as ff

import plotly.express as px

init_notebook_mode()  
df = pd.read_csv('/kaggle/input/golden-globe-awards/golden_globe_awards.csv')

df
winners = df[df['win']==True]

winners = winners.groupby(['film'])['win'].count().reset_index().sort_values('win', ascending=True)

winners = winners[winners['win']>=5]

fig = px.bar(winners, x="win", y="film", orientation='h', title='Films with at least 5 awards')

fig.show()
first_year_award = df[['category','year_award']].groupby('category').first().reset_index()

last_year_award = df[['category','year_award']].groupby('category').last().reset_index()

first_year_award.columns = ['category', 'first_year']

last_year_award.columns = ['category', 'last_year']

dates_df = pd.merge(first_year_award, last_year_award, on='category', how='inner')

dates_df.head(20)

fig = px.bar(dates_df[dates_df['last_year']==2020], x="first_year", y="category", orientation='h', title='Start year for awards that are present in 2020')

fig.show()
def pie_count(data, field="Nationality", percent_limit=0.5, title="Number of nominees by "):

    

    title += field

    data[field] = data[field].fillna('NA')

    data = data[field].value_counts().to_frame()



    total = data[field].sum()

    data['percentage'] = 100 * data[field]/total    



    percent_limit = percent_limit

    otherdata = data[data['percentage'] < percent_limit] 

    others = otherdata['percentage'].sum()  

    maindata = data[data['percentage'] >= percent_limit]



    data = maindata

    other_label = "Others(<" + str(percent_limit) + "% each)"           # Create new label

    data.loc[other_label] = pd.Series({field:otherdata[field].sum()}) 

    

    labels = data.index.tolist()   

    datavals = data[field].tolist()

    

    trace=go.Pie(labels=labels,values=datavals)



    layout = go.Layout(

        title = title,

        height=900,

        width =1500

        )

    

    fig = go.Figure(data=[trace], layout=layout)

    iplot(fig)



pie_count(df, 'category')
year_award = df['year_award'].value_counts().reset_index()

year_award.columns = ['year', 'award_count']

fig = px.bar(year_award, x="year", y="award_count", orientation='v', title='Number of awards for every year')

fig.show()
nominees = df['nominee'].value_counts().reset_index().sort_values('nominee', ascending=True).tail(20)

fig = px.bar(nominees, x="nominee", y="index", orientation='h', title='Top nominees during the history')

fig.show()
win = df[df['win']==True]['nominee'].value_counts().reset_index().sort_values('nominee', ascending=True).tail(20)

fig = px.bar(win, x="nominee", y="index", orientation='h', title='Top nominees winners during the history')

fig.show()