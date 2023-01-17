# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

countries_iso = pd.read_csv("/kaggle/input/countries-of-the-world-iso-codes-and-population/country_codes_2020.csv")

countries=pd.read_csv("/kaggle/input/countries-of-the-world-iso-codes-and-population/countries_by_population_2019.csv")
df.head()
df.info()
df.drop('Last Update',axis=1,inplace = True)



#Making new column of Active Cases in dataset

df['Active'] = df['Confirmed']-df['Deaths']-df['Recovered']



#Converting string Date time into Python Date time object

df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])

df['Confirmed']=df['Confirmed'].astype('int')

df['Deaths']=df['Deaths'].astype('int')

df['Recovered']=df['Recovered'].astype('int')

df['Active']=df['Active'].astype('int')



#Renaming columns

df.rename(columns = {'ObservationDate':'Date','Recovered':'Cured','Province/State':'State','Country/Region':'Country'}, inplace = True)



#Replacing null values

df["State"].fillna("No State",inplace=True) 



data_df=df.groupby('Date').sum()

data_df.reset_index(inplace=True)



#Making new columns of Recovery and Death Rate.

data_df['Recovery_Rate']=data_df['Cured']/data_df['Confirmed']

data_df['Death_Rate']=data_df['Deaths']/data_df['Confirmed']
df_updated = df[df['Date'] == max(df['Date'])].reset_index()

df_updated_grouped = df_updated.groupby('Country')['Confirmed', 'Deaths', 'Cured','Active'].sum().reset_index()

temp = df_updated.groupby(['Country', 'State'])['Confirmed', 'Cured', 'Deaths','Active'].max()

temp.style.background_gradient(cmap='Pastel1_r')
temp.sum()
temp_f = df_updated_grouped[['Country', 'Confirmed','Cured','Deaths','Active']]

temp_f = temp_f.sort_values(by='Confirmed', ascending=False)

temp_f = temp_f.reset_index(drop=True)

temp_f.style.background_gradient(cmap='Pastel1_r')
temp10=temp_f.head(10)

temp_fc=list(temp10['Country'])

plt.figure(figsize=(10,5))

sns.barplot(x=temp_fc,y=temp10['Confirmed'])

plt.grid(True)

plt.title('Confirmed Cases',size = 20)

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Country",fontsize = 18)

plt.ylabel('Confirmed in million',fontsize = 18)

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x=temp_fc,y=temp10['Cured'])

plt.grid(True)

plt.title('Cured Cases',size = 20)

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Country",fontsize = 18)

plt.ylabel('Cured',fontsize = 18)

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x=temp_fc,y=temp10['Deaths'])

plt.grid(True)

plt.title('Death Cases',size = 20)

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Country",fontsize = 18)

plt.ylabel('Deaths',fontsize = 18)

plt.legend(loc=0)

plt.show()
plt.figure(figsize=(10,5))

sns.barplot(x=temp_fc,y=temp10['Active'])

plt.grid(True)

plt.title('Active Cases',size = 20)

plt.tick_params(size=5,labelsize = 13)

plt.xlabel("Country",fontsize = 18)

plt.ylabel('Active in million',fontsize = 18)

plt.legend(loc=0)

plt.show()
f, ax = plt.subplots(figsize=(10, 5))

plt.tick_params(size=5,labelsize = 13)

plt.ylabel('Country',fontsize = 18)

bar1=sns.barplot(x="Confirmed",y="Country",data=temp10,

            label="Confirmed", color="#0065b3")



bar2=sns.barplot(x="Cured", y="Country", data=temp10,

            label="Cured", color="#03ff39")



bar3=sns.barplot(x="Deaths", y="Country", data=temp10,

            label="Deaths", color="red")



ax.legend(loc=4, ncol = 1)

plt.xlabel("Total Cases",fontsize = 18)

plt.show()
world_province_cases=temp10[['Confirmed','Cured','Active','Deaths','Country']].groupby('Country').max().sort_values('Confirmed',ascending=False)

world_province_cases.plot(kind='bar',width=0.95,colormap='rainbow',figsize=(10,5),fontsize = 13)

plt.grid(True)

plt.show()
fig = plt.figure(figsize=(7,7))

conf_per_state = temp10.groupby('Country')['Confirmed'].max().sort_values(ascending=False)



def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_state.plot(kind="pie",title='Percentage of confirmed cases per country',autopct=absolute_value)

plt.legend(loc=1, ncol = 7)

plt.show()
fig = plt.figure(figsize=(7,7))

conf_per_state = temp10.groupby('Country')['Cured'].max().sort_values(ascending=False)



def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_state.plot(kind="pie",title='Percentage of cured cases per country',autopct=absolute_value)

plt.legend(loc=1, ncol = 7)

plt.show()
fig = plt.figure(figsize=(7,7))

conf_per_state = temp10.groupby('Country')['Deaths'].max().sort_values(ascending=False)



def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_state.plot(kind="pie",title='Percentage of death cases per country',autopct=absolute_value)

plt.legend(loc=1, ncol = 7)

plt.show()
fig = plt.figure(figsize=(7,7))

conf_per_state = temp10.groupby('Country')['Active'].max().sort_values(ascending=False)



def absolute_value(val):

    a  = val

    return (np.round(a,2))

conf_per_state.plot(kind="pie",title='Percentage of active cases per country',autopct=absolute_value)

plt.legend(loc=1, ncol = 7)

plt.show()
plt.figure(figsize= (14,8))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Total cases in millions',fontsize = 20)

plt.title("Total Confirmed, Cured, Deaths and Active cases in World" , fontsize = 20)



ax1 = plt.plot_date(data=data_df,y= 'Confirmed',x= 'Date',label = 'Confirmed',linestyle ='-',color = 'b')

ax2 = plt.plot_date(data=data_df,y= 'Cured',x= 'Date',label = 'Cured',linestyle ='-',color = 'g')

ax3 = plt.plot_date(data=data_df,y= 'Deaths',x= 'Date',label = 'Death',linestyle ='-',color = 'r')

ax4 = plt.plot_date(data=data_df,y= 'Active',x= 'Date',label = 'Active',linestyle ='-',color = 'y')



plt.legend();
plt.figure(figsize= (14,8))

plt.xticks(rotation = 90 ,fontsize = 10)

plt.yticks(fontsize = 10)

plt.xlabel("Dates",fontsize = 20)

plt.ylabel('Rate',fontsize = 20)

plt.title("Recovery and Death Rate in World" , fontsize = 20)



ax1 = plt.plot_date(data=data_df,y= 'Recovery_Rate',x= 'Date',label = 'Recovery_rate',linestyle ='-',color = 'g')

ax2 = plt.plot_date(data=data_df,y= 'Death_Rate',x= 'Date',label = 'Death_rate',linestyle ='-',color = 'r')



plt.legend();
import plotly.graph_objects as go



# Create figure

fig = go.Figure()



fig.add_trace(

    go.Scatter(x=list(data_df.Date), y=list(data_df.Confirmed)))



# Set title

fig.update_layout(

    title_text="Confirmed cases over time"

)



# Add range slider

fig.update_layout(

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    )

)



fig.show()
import plotly.graph_objects as go



# Create figure

fig = go.Figure()



fig.add_trace(

    go.Scatter(x=list(data_df.Date), y=list(data_df.Cured)))



# Set title

fig.update_layout(

    title_text="Cured cases over time"

)



# Add range slider

fig.update_layout(

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    )

)



fig.show()
import plotly.graph_objects as go



# Create figure

fig = go.Figure()



fig.add_trace(

    go.Scatter(x=list(data_df.Date), y=list(data_df.Deaths)))



# Set title

fig.update_layout(

    title_text="Death cases over time"

)



# Add range slider

fig.update_layout(

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    )

)



fig.show()
import plotly.graph_objects as go



# Create figure

fig = go.Figure()



fig.add_trace(

    go.Scatter(x=list(data_df.Date), y=list(data_df.Active)))



# Set title

fig.update_layout(

    title_text="Active cases over time"

)



# Add range slider

fig.update_layout(

    xaxis=dict(

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            visible=True

        ),

        type="date"

    )

)



fig.show()
cols_to_drop = ['Rank', 'pop2018','GrowthRate', 'area', 'Density']

countries = countries.drop(columns = cols_to_drop)





countries = countries.merge(countries_iso[['name', 'cca3']], on = ['name'], how = "left")



cols_to_rename = {'name': 'Country', 'pop2019': 'Population', 'cca3': 'ISO'}

countries = countries.rename(columns = cols_to_rename)



#fixing the most important mismatches

countries_to_rename = {'US': 'United States',\

                       'Mainland China': 'China',\

                       'UK': 'United Kingdom',\

                       'Congo (Kinshasa)': 'DR Congo',\

                       'North Macedonia': 'Macedonia',\

                       'Republic of Ireland': 'Ireland',\

                       'Congo (Brazzaville)': 'Republic of the Congo'}



temp_f['Country'] = temp_f['Country'].replace(countries_to_rename)



temp_map = temp_f.merge(countries, on = "Country", how = "left")
temp_map.head()

fig = px.choropleth(temp_map, locations="ISO",

                    color="Confirmed",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.YlOrRd)



layout = go.Layout(

    title=go.layout.Title(

        text="Corona confirmed cases",

        x=0.5

    ),

    font=dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)



fig.show()
fig = px.choropleth(temp_map, locations="ISO",

                    color="Cured",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.YlGn)



layout = go.Layout(

    title=go.layout.Title(

        text="Corona cured cases",

        x=0.5

    ),

    font=dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)



fig.show()
fig = px.choropleth(temp_map, locations="ISO",

                    color="Deaths",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.YlOrRd)



layout = go.Layout(

    title=go.layout.Title(

        text="Corona death cases",

        x=0.5

    ),

    font=dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)



fig.show()
fig = px.choropleth(temp_map, locations="ISO",

                    color="Active",

                    hover_name="Country",

                    color_continuous_scale=px.colors.sequential.YlOrRd)



layout = go.Layout(

    title=go.layout.Title(

        text="Corona active cases",

        x=0.5

    ),

    font=dict(size=14),

    width = 750,

    height = 350,

    margin=dict(l=0,r=0,b=0,t=30)

)



fig.update_layout(layout)



fig.show()