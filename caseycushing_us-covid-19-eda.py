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


US_data_nyt = pd.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')



US_data_nyt = US_data_nyt.sort_values(by=['state'],ascending=True).reset_index(drop=True)



US_data_nyt
US_data_nyt_state = US_data_nyt.groupby(['state','county','date'])['deaths','cases'].apply(lambda x: x.sum())



US_data_nyt_state = US_data_nyt_state.reset_index()



US_data_nyt_state = US_data_nyt_state.sort_values(by='date',ascending=False)



US_data_nyt_state = US_data_nyt_state.reset_index(drop=True)



US_data_nyt_state
state_date = US_data_nyt_state.groupby(['state','date'])['deaths','cases'].apply(lambda x: x.sum())



state_date = state_date.reset_index()



state_date
state_total = state_date.groupby('state')['cases'].sum()



state_total = state_total.reset_index()



state_total = state_total.sort_values(by=['cases'],ascending=False)



state_total
import plotly.express as px

fig = px.line(state_date,x='date',y='deaths',color='state')



fig.show()
fig = px.line(state_date,x='date',y='cases',color='state')



fig.show()
state_date = state_date.query('state != "New York"')



state_date
fig = px.line(state_date,x='date',y='deaths',color='state')



fig.show()
#US_data_nyt = US_data_nyt_state.query('state != "New York"')



state_date = state_date.query('deaths >= 20')



fig = px.scatter(state_date,x='date',y='deaths',color='state',size='deaths')



fig.show()




or_county = US_data_nyt_state.groupby(['state','county'])['deaths'].max()









or_county = or_county.reset_index()



or_county = or_county.sort_values(by='deaths',ascending=False)



or_county = or_county.reset_index(drop=True)



                                  

or_county
or_county = or_county.query('state == "Oregon"')



or_county
import plotly.express as px
fig = px.scatter(or_county,x=or_county['county'],y=or_county['deaths'],size=or_county['deaths'],color=or_county['deaths'])



fig.show()
#US_data_high_death = US_data_nyt.query('deaths >= 100')



state_date = state_date.sort_values(by=['deaths'],ascending=False).reset_index(drop=True)



fig = px.scatter(state_date,x=state_date['state'],y=state_date['deaths'],size=state_date['deaths'],color=state_date['deaths'])



fig.show()
#state_date = state_date.query('cases >= 2000')



state_date = state_date.sort_values(by=['cases'],ascending=False).reset_index(drop=True)



fig = px.scatter(state_date,x=state_date['state'],y=state_date['cases'],size=state_date['deaths'],color=state_date['cases'])



fig.show()
state_date
US_data_nyt_date = US_data_nyt.loc[:,['date','state','deaths']]



US_data_nyt_date = US_data_nyt_date.sort_values(by='deaths',ascending = False)



US_data_nyt_date = pd.DataFrame(data=US_data_nyt_date).reset_index(drop=True)





US_data_nyt_date
fig = px.scatter(state_total,x='state',y='cases',color='state',size='cases')



fig.show()