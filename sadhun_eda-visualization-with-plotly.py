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
# import the necessary libraries

import numpy as np 

import pandas as pd 

import os



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()

import pycountry

import plotly.express as px

from plotly.offline import init_notebook_mode, iplot 

import plotly.graph_objs as go

import plotly.offline as py

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot



!pip install pywaffle

from pywaffle import Waffle



py.init_notebook_mode(connected=True)

import folium 

from folium import plugins

plt.style.use("fivethirtyeight")# for pretty graphs



# Increase the default plot size and set the color scheme

plt.rcParams['figure.figsize'] = 8, 5

#plt.rcParams['image.cmap'] = 'viridis'



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Disable warnings 

import warnings

warnings.filterwarnings('ignore')

df= pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df.head(4)
#Total number of city and resort bookings

colors = ['#1f77b4', '#17becf']

city,resort=df['hotel'].value_counts().values.tolist()

fig = go.Figure(data=[go.Pie(labels=['City Hotel','Resort Hotel'],

                             values= [city,resort],hole =.3)])

                          



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))

fig.show()
#Confirmed Hotel bookings. excluded the cancellations



city_con=(df[df['hotel']=='City Hotel'].iloc[:,1].count())-(df[df['hotel']=='City Hotel'].iloc[:,1].sum())

resort_con=(df[df['hotel']=='Resort Hotel'].iloc[:,1].count())-(df[df['hotel']=='Resort Hotel'].iloc[:,1].sum())



#Total number of city and resort bookings

fig = go.Figure(data=[go.Pie(labels=['Conf City Hotel','Conf Resort Hotel'],

                             values= [city_con,resort_con],hole =.3)])

                          



fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))

fig.show()
Mnth=df.groupby('arrival_date_month').size().reset_index()

Mnth.columns=['Month','Bookings']

def highlight_max(s):

    is_max = s == s.max()

    return ['background-color: pink' if v else '' for v in is_max]



Mnth.style.apply(highlight_max,subset=['Bookings'])

fig = px.bar(Mnth.sort_values('Bookings', ascending=False).sort_values('Bookings', ascending=True), 

             x="Bookings", y="Month", 

             title='Total Active Bookings in a month', 

             text='Bookings', 

             orientation='h', 

             width=1000, height=700, range_x = [0, max(Mnth['Bookings'])])

fig.update_traces(marker_color='#46cdcf', opacity=0.8, textposition='inside')



fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
#Top 20 Country origins where most of the bookings are made

Cntry=df.groupby('country').size().reset_index()

Cntry.columns=['Code','Count']

Cntry=Cntry.sort_values('Count',ascending=False)

Cntry.head(20).style.background_gradient(cmap='Reds')
#Waffle representation of previous cancellations and previous bookings

df_can = pd.DataFrame([df['previous_cancellations'].sum(),df['previous_bookings_not_canceled'].sum()],columns=['Cases'])

df_can.index=['can','book']

df_can



fig = plt.figure(

    FigureClass=Waffle, 

    rows=15,

    values=df_can['Cases'],

    labels=list(df_can.index),

    figsize=(15,30),

    legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)}

)
seg=df.groupby('market_segment').size().reset_index()

seg.columns=['market_segment','count']

seg=seg.sort_values('count',ascending=False)

seg



dis=df.groupby('distribution_channel').size().reset_index()

dis.columns=['distribution_channel','count']

dis=dis.sort_values('count',ascending=False)



dis
#Bar chart for market segment vs distribution channel

import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Market Segment","Distribution Channel"))



#temp = df.sort_values('Total Confirmed cases (Indian National)', ascending=False).sort_values('Total Confirmed cases (Indian National)', ascending=False)



fig.add_trace(go.Bar( y=seg['count'], x=seg['market_segment'],  

                     marker=dict(color=seg['count'], coloraxis="coloraxis")),

              1, 1)





fig.add_trace(go.Bar( y=dis['count'], x=dis['distribution_channel'],  

                     marker=dict(color=dis['count'], coloraxis="coloraxis")),

              1, 2)

fig.update_layout(coloraxis=dict(colorscale='rdbu'), showlegend=False,title_text="Market vs Distribution",plot_bgcolor='rgb(250, 242, 242)')

fig.show()

htl=pd.DataFrame(df['hotel'].value_counts()).reset_index()

htl.columns=['Type','Total Bookings']

htl['Cancelled']=[df[df['hotel']=='City Hotel'].iloc[:,1].sum(),df[df['hotel']=='Resort Hotel'].iloc[:,1].sum()]

htl
f, ax = plt.subplots(figsize=(10, 5))

sns.set_color_codes("pastel")

sns.barplot(x="Total Bookings", y="Type", data=htl,

            label="Total", color="b")



sns.set_color_codes("muted")

sns.barplot(x="Cancelled", y="Type", data=htl,

            label="Cancellations", color="r")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 80000), ylabel="",

       xlabel="Cancellations")

sns.despine(left=True, bottom=True)
bkng=pd.DataFrame(df['hotel'].value_counts()).reset_index()

bkng.columns=['Type','Total Bookings']

bkng['Confirmed']=[(df[df['hotel']=='City Hotel'].iloc[:,1].count())-(df[df['hotel']=='City Hotel'].iloc[:,1].sum()),

                   (df[df['hotel']=='Resort Hotel'].iloc[:,1].count())-(df[df['hotel']=='Resort Hotel'].iloc[:,1].sum())]



f, ax = plt.subplots(figsize=(10, 5))

sns.set_color_codes("pastel")

sns.barplot(x="Total Bookings", y="Type", data=bkng,

            label="Total", color="b")



sns.set_color_codes("muted")

sns.barplot(x="Confirmed", y="Type", data=bkng,

            label="Confirmed bookings", color="g")





# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 80000), ylabel="",

       xlabel="Bookings")

sns.despine(left=True, bottom=True)