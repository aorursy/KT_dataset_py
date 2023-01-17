# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory







from bokeh.plotting import output_notebook, figure, show

from bokeh.models import ColumnDataSource, Div, Select, Button, ColorBar, CustomJS

from bokeh.layouts import row, column, layout

from bokeh.transform import cumsum, linear_cmap

from bokeh.palettes import Blues8, Spectral3

from bokeh.plotting import figure, output_file, show



output_notebook()



# Visualisation libraries

import matplotlib.pyplot as plt

%matplotlib inline



import folium 

from folium import plugins

plt.style.use("fivethirtyeight")# for pretty graphs











from plotly.offline import iplot

from plotly import tools

import plotly.graph_objects as go

import plotly.express as px

import plotly.offline as py

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df.head()
temp = train_df.groupby(['Date', 'Country/Region'])['ConfirmedCases'].sum().reset_index()

temp['Date'] = pd.to_datetime(temp['Date'])

temp['Date'] = temp['Date'].dt.strftime('%m/%d/%Y')

temp['size'] = temp['ConfirmedCases'].pow(0.3) * 3.5



fig = px.scatter_geo(temp, locations="Country/Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country/Region", 

                     range_color=[1,100],

                     projection="natural earth", animation_frame="Date", 

                     title='COVID-19: Cases Over Time', color_continuous_scale="greens")

fig.show()
Country=pd.DataFrame()

#temp = train_df.groupby(["Country/Region"])["ConfirmedCases"].sum().reset_index()

temp = train_df.loc[train_df["Date"]==train_df["Date"][len(train_df)-1]].groupby(['Country/Region'])["ConfirmedCases"].sum().reset_index()

Country['Name']=temp["Country/Region"]

Country['Values']=temp["ConfirmedCases"]



fig = px.choropleth(Country, locations='Name',

                    locationmode='country names',

                    color="Values")

fig.update_layout(title="Corona spread on 21-03-2020")

fig.show()
Disease_through_Country = pd.DataFrame()

Disease_through_Country = train_df.groupby(["Country/Region"]).sum().reset_index()

Disease_through_Country = Disease_through_Country.drop(['Lat','Long'],axis=1)



Names = ["ConfirmedCases","Fatalities"]

for i in Names:

    Disease_through_Country[i+"_percentage"] = Disease_through_Country[i]/Disease_through_Country[Names].sum(axis=1)*100

    Disease_through_Country[i+"_angle"] = Disease_through_Country[i+"_percentage"]/100 * 2*np.pi

    

Disease_through_Country_plot = pd.DataFrame({'class': ["ConfirmedCases","Fatalities"],

                                              'percent': [float('nan'), float('nan')],

                                              'angle': [float('nan'), float('nan')],

                                              'color': [ '#718dbf', '#e84d60']})

Disease_through_Country_plot
# Create the ColumnDataSource objects "s2" and "s2_plot"

s2 = ColumnDataSource(Disease_through_Country)

s2_plot = ColumnDataSource(Disease_through_Country_plot)



# Create the Figure object "p2"

p2 = figure(plot_width=475, plot_height=550, y_range=(-0.5, 0.7),toolbar_location=None, tools=['hover'], tooltips='@percent{0.0}%')



# Add circular sectors to "p2"

p2.wedge(x=0, y=0, radius=0.8, source=s2_plot,start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),fill_color='color', line_color=None, legend='class')



# Change parameters of "p2"

p2.axis.visible = False

p2.grid.grid_line_color = None

p2.legend.orientation = 'horizontal'

p2.legend.location = 'top_center'



# Create the custom JavaScript callback

callback2 = CustomJS(args=dict(s2=s2, s2_plot=s2_plot), code='''

    var ang = ['ConfirmedCases_angle', 'Fatalities_angle'];

    var per = ['ConfirmedCases_percentage',  'Fatalities_percentage'];

    if (cb_obj.value != 'Please choose...') {

        var disease = s2.data['Country/Region'];

        var ind = disease.indexOf(cb_obj.value);

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = s2.data[ang[i]][ind];

            s2_plot.data['percent'][i] = s2.data[per[i]][ind];

            

        }

    }

    else {

        for (var i = 0; i < ang.length; i++) {

            s2_plot.data['angle'][i] = undefined;

            s2_plot.data['percent'][i] = undefined;

        }



    }

    s2_plot.change.emit();

''')



# When changing the value of the dropdown menu execute "callback2"

options = ['Please choose...'] + list(s2.data['Country/Region'])

select = Select(title='Country ', value=options[0], options=options)

select.js_on_change('value', callback2)



# Display "select" and "p2" as a column

show(column(select, p2))
Data = train_df.groupby("Date").sum().reset_index()

Data["Date"]= pd.to_datetime(Data["Date"])

source = ColumnDataSource(Data)

p = figure(x_axis_type='datetime')







p.line(x='Date', y='ConfirmedCases', line_width=2, source=source, legend_label='Confirmed Corona Cases')

p.line(x='Date', y='Fatalities', line_width=2, source=source, color=Spectral3[2], legend_label='Death by Corona')



p.yaxis.axis_label = 'Activity of Corona period of time'

show(p)
train_df["Date"] = pd.to_datetime(train_df["Date"])
China_cases = train_df.loc[train_df["Country/Region"]=="China"].groupby("Date")["ConfirmedCases"].sum().reset_index()

Italy_cases = train_df.loc[train_df["Country/Region"]=="Italy"].groupby("Date")["ConfirmedCases"].sum().reset_index()

Iran_cases = train_df.loc[train_df["Country/Region"]=="Iran"].groupby("Date")["ConfirmedCases"].sum().reset_index()

Usa_cases = train_df.loc[train_df["Country/Region"]=="Usa"].groupby("Date")["ConfirmedCases"].sum().reset_index()

Colombia_cases = train_df.loc[train_df["Country/Region"]=="Colombia"].groupby("Date")["ConfirmedCases"].sum().reset_index()





fig = go.Figure()



fig.add_trace(go.Scatter(x=China_cases.Date, y=China_cases['ConfirmedCases'], name="Cases in China",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=Italy_cases.Date, y=Italy_cases['ConfirmedCases'], name="Cases in Itlay",

                         line_color='red'))



fig.add_trace(go.Scatter(x=Iran_cases.Date, y=Iran_cases['ConfirmedCases'], name="Cases in Iran",

                         line_color='green'))



fig.add_trace(go.Scatter(x=Usa_cases.Date, y=Usa_cases['ConfirmedCases'], name="Cases in Usa",

                         line_color='yellow'))



fig.add_trace(go.Scatter(x=Colombia_cases.Date, y=Colombia_cases['ConfirmedCases'], name="Cases in Colombia",

                         line_color='purple'))



fig.update_layout(title_text='Spread of Corona over a period of Time',

                  xaxis_rangeslider_visible=True)

fig.show()
China_cases = train_df.loc[train_df["Country/Region"]=="China"].groupby("Date")["Fatalities"].sum().reset_index()

Italy_cases = train_df.loc[train_df["Country/Region"]=="Italy"].groupby("Date")["Fatalities"].sum().reset_index()

Iran_cases = train_df.loc[train_df["Country/Region"]=="Iran"].groupby("Date")["Fatalities"].sum().reset_index()

Usa_cases = train_df.loc[train_df["Country/Region"]=="Usa"].groupby("Date")["Fatalities"].sum().reset_index()

Colombia_cases = train_df.loc[train_df["Country/Region"]=="Colombia"].groupby("Date")["Fatalities"].sum().reset_index()





fig = go.Figure()



fig.add_trace(go.Scatter(x=China_cases.Date, y=China_cases['Fatalities'], name="Fatalities in China",

                         line_color='deepskyblue'))



fig.add_trace(go.Scatter(x=Italy_cases.Date, y=Italy_cases['Fatalities'], name="Fatalities in Itlay",

                         line_color='red'))



fig.add_trace(go.Scatter(x=Iran_cases.Date, y=Iran_cases['Fatalities'], name="Fatalities in Iran",

                         line_color='green'))



fig.add_trace(go.Scatter(x=Usa_cases.Date, y=Usa_cases['Fatalities'], name="Fatalities in Usa",

                         line_color='yellow'))



fig.add_trace(go.Scatter(x=Colombia_cases.Date, y=Colombia_cases['Fatalities'], name="Fatalities in Colombia",

                         line_color='purple'))



fig.update_layout(title_text='Fatality through Corona over a period of Time',

                  xaxis_rangeslider_visible=True)

fig.show()
import math

def Survival(Country):

    Sx = [] 

    d = 1

    Ld = 0

    temp_ = train_df.loc[train_df["Country/Region"]==Country]

    temp = temp_.groupby(['Date'])['Fatalities','ConfirmedCases'].sum().reset_index()

    temp["Survival Probability"] = 0

    temp["Hazard Rate"] = 0

    Hr = []

    for i in range(len(temp)):

        delta = 1

        d = temp["Fatalities"][i]

        n = temp["ConfirmedCases"][i]

        L = Ld + math.pow((d/n),delta)

        S = math.exp(-L)

        Hr.append(L)

        Sx.append(S)

        d= temp["Fatalities"][i]

        Ld = 0

    temp["Survival Probability"] = Sx

    temp["Hazard Rate"] = Hr

    return temp

    

China_df = Survival("China")

Italy_df = Survival("Italy")

Iran_df = Survival("Iran")

Usa_df = Survival("USA")

Colombia_df = Survival("Colombia")
China_df.head()
fig = go.Figure()





fig.add_trace(go.Scatter(x=Italy_df.Date, y=Italy_df['Survival Probability'], name="Italy",

                         line_color='red'))



fig.add_trace(go.Scatter(x=Iran_df.Date, y=Iran_df['Survival Probability'], name="Iran",

                         line_color='blue'))



fig.add_trace(go.Scatter(x=Usa_df.Date, y=Usa_df['Survival Probability'], name="Usa",

                         line_color='green'))



fig.add_trace(go.Scatter(x=China_df.Date, y=China_df['Survival Probability'], name="China",

                         line_color='black'))



fig.add_trace(go.Scatter(x=Colombia_df.Date, y=Colombia_df['Survival Probability'], name="Colombia",

                         line_color='purple'))



fig.update_layout(title_text='Survival Probability Corona over a period of Time',

                  xaxis_rangeslider_visible=True)



fig.show()
temp_df = train_df.loc[train_df["Date"]=="2020-03-20"].groupby("Country/Region")["ConfirmedCases","Fatalities"].sum().reset_index()

temp=pd.DataFrame()

temp["Index"] = ["Korea,South","Colombia","Iran","Italy","China","Others"]

t = temp_df.sort_values(by="ConfirmedCases").tail()["ConfirmedCases"].values

values = []

for i in range(0,5):

    values.append(t[i])

values.append(sum(temp_df.loc[~temp_df["Country/Region"].isin(temp["Index"])]["ConfirmedCases"]))

temp["Values"]=values



fig = go.Figure(data=[go.Pie(labels=temp["Index"], values=temp["Values"],hole=0.2)])

fig.show()
Colombia_df.head()
temp = train_df.loc[(train_df["Country/Region"]=="Colombia") & (train_df["Date"]=="2020-03-20")].groupby(["Province/State","Lat","Long"])["ConfirmedCases"].sum().reset_index()

map = folium.Map(location=[34, 100], zoom_start=3.5,tiles='Stamen Toner')



for lat, lon, value, name in zip(temp['Lat'], temp['Long'], temp['ConfirmedCases'], temp["Province/State"]):

    folium.CircleMarker([lat, lon],

                        radius=value*0.007,

                        popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Active Cases</strong>: ' + str(value) + '<br>'),

                        color='purple',

                        

                        fill_color='purple',

                        fill_opacity=0.3 ).add_to(map)

map
temp = train_df.loc[(train_df["Country/Region"]=="Colombia")].groupby(["Date"])["ConfirmedCases"].sum().reset_index()

fig = px.bar(temp, x='Date', y='ConfirmedCases',

             hover_data=['ConfirmedCases'], color='ConfirmedCases',

             labels={'pop':'Total Number of confirmed Cases'}, height=400)

fig.show()
from plotly.subplots import make_subplots

import plotly.graph_objects as go



Sk_df = train_df.loc[train_df["Country/Region"]=="Korea, South"].groupby("Date").sum().reset_index()

Colombia_df = train_df.loc[train_df["Country/Region"]=="Korea, South"].groupby("Date").sum().reset_index()



fig = make_subplots(rows=1, cols=2, shared_yaxes=True)



fig.add_trace(go.Bar(x=Sk_df.Date, y=Sk_df.ConfirmedCases, name="South Korea"),1, 1)



fig.add_trace(go.Bar(x=Colombia_df.Date, y=Iran_df.ConfirmedCases, name ="Colombia"),1, 2)



#fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

fig.update_layout(

    title_text="Disease Spread Pattern in South Korea and Colombia")

fig.show()