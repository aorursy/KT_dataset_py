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
#Import the necessary libraries 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython import display

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Data importing from Kaggle COVID19 week 1
df1=pd.read_csv("/kaggle/input/covid19dataa/covid_19_clean_complete.csv",parse_dates=['Date'])
# Data importing from Kaggle COVID19 week 3
df2=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")
# Examining the week 1 dataset
df1.info()
# Examining the week 3 dataset
df2.info()
#Examining the period of which the data (week 1) is collected
a = df1.Date.value_counts().sort_index()
print('The first date recorded in the 1st week dataset is:',a.index[0])
print('The last date recorded in the 1st week dataset is:',a.index[-1])
#Data Cleaning

#Renaming the column to simpler for easy usage
df1.rename(columns={'Date': 'date', 
                     'Province/State':'state',
                     'Country/Region':'country',
                     'Lat':'lat', 'Long':'long',
                     'Confirmed': 'confirmed',
                     'Deaths':'deaths',
                     'Recovered':'recovered'
                    }, inplace=True)

# Active Case = confirmed - deaths - recovered
df1['active'] = df1['confirmed'] - df1['deaths'] - df1['recovered']
#Data Cleaning

#Renaming the column to simpler for easy usage
df2.rename(columns={'Date': 'date', 
                     'Province_State':'state',
                     'Country_Region':'country',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                  }, inplace=True)

#create whole figure
def create_figure(width,height,Title, Title_font, Title_bold, 
                  left_adjust,bottom_adjust, right_adjust,top_adjust,wspace_adjust,hspace_adjust):
    fig = plt.figure(figsize=(width,height))
    plt.title(Title, fontsize=Title_font, fontweight=Title_bold)            
    plt.subplots_adjust(left=left_adjust, bottom=bottom_adjust, right=right_adjust, top=top_adjust, 
                        wspace=wspace_adjust, hspace=hspace_adjust)    
    return fig,plt

#set the fig x and y label
def set_fig_xy_label(plt, xlabel, ylabel, x_font, y_font):
    plt.xlabel(xlabel, fontsize = x_font)
    plt.ylabel(ylabel, fontsize = y_font)    
    
    return plt

#creating scatter plot
def display_scatter_geo(data, location_label, locmode_label,color_label, hover_label,size_label,proj_label,title_caption):
    return px.scatter_geo(data, locations=location_label,
                        locationmode=locmode_label, color=color_label,
                        hover_name=hover_label, size=size_label,
                        projection=proj_label,title=title_caption)  

#creating choropleth

def display_choropleth(data,location_label,locmode_label,color_label,hover_label,range_color, scale_color, title_caption):
    return px.choropleth(data, locations=location_label,locationmode=locmode_label, color=color_label,
                        hover_name=hover_label, range_color=range_color,
                    color_continuous_scale=scale_color,title=title_caption) 

# draw barplot
def display_barplot(ax, y_data, x_data, ax_title, xlabel, ylabel, title_font, 
                    xlabel_font, ylabel_font,xtick_font, ytick_font, val_font):
    sns.barplot(y=y_data, x=x_data)
    ax.set_title(ax_title,fontsize=title_font)
    ax.set_xlabel(xlabel, fontsize=xlabel_font)
    ax.set_ylabel(ylabel, fontsize=ylabel_font)     
    
    ax.tick_params(axis="x", labelsize=xtick_font)
    ax.tick_params(axis="y", labelsize=ytick_font)
    
    for i, (value, name) in enumerate(zip(x_data,y_data)):
        ax.text(value, i-.05, f'{value:,.0f}', size=val_font, ha='left', va='center')    
    
    return ax

# create widget
def create_widget_Text_w_df(label,data):
    widget1 = widgets.Output()

    # render in output widgets
    with widget1:        
        display.display(label)
        display.display(data)
    
    return widget1
#Examining the inital worldwide spread of COVID-19 (22-01-2020 to 29-03-2020)

df1['date'] = pd.to_datetime(df1['date'])
df1['date'] = df1['date'].dt.strftime('%m/%d/%Y')
df1 = df1.fillna('-')

fig1 = px.density_mapbox(df1, lat='lat', lon='long', z='confirmed', radius=20,zoom=1, hover_data=["country",'state',"confirmed"],
                        mapbox_style="carto-positron", animation_frame = 'date', range_color= [0, 1000],title='Spread of COVID-19(22-01-2020 to 29-03-2020)')
fig1.update_layout(margin={"r":0,"t":30,"l":0,"b":0})

#Confired cases around the word (22-01-2020 to 29-03-2020)
top = df1[df1['date'] == df1['date'].max()]
world1 = top.groupby('country')['confirmed','deaths','date','recovered'].sum().reset_index()
world1.head()
#Confired cases around the word (22-01-2020 to 05-04-2020)
top2 = df2[df2['date'] == df2['date'].max()]
world2 = top2.groupby('country')['confirmed','deaths','date'].sum().reset_index()
world2.head()
fig2 = display_choropleth(world1, "country", "country names", "confirmed","country",[1,10000],"Peach", 'Countries with Confirmed Cases(22-01-2020 to 29-03-2020)')
fig2
fig3 = display_choropleth(world2, "country", "country names", "confirmed","country",[1,10000],"Peach", 'Countries with Confirmed Cases(22-01-2020 to 05-04-2020)')
fig3
## Set the data
Fig4 = df1[['country', 'date', 'confirmed', 'deaths', 'recovered']]
Fig4 = Fig4.groupby(['country', 'date']).sum().reset_index()
Fig4.sort_values('date', ascending=True, inplace=True)

## Visualize the data
px.scatter(Fig4, 
           x="confirmed", 
           y="recovered", 
           animation_frame="date", 
           animation_group="country", 
           height = 800,
           size="confirmed", 
           color="recovered", 
           hover_name="country", 
           color_continuous_scale='Reds',
           title = 'Correlation between confirmed cases and recovery cases from COVID-19 (22-01-2020 to 29-03-2020 )',
           range_color=[0,5000],
           log_x=True,
           text ='country',
           size_max=100, 
           range_x=[100,300000], 
           range_y=[-9000,100000])
## Set the data
Fig5 = df1[['country', 'date', 'confirmed', 'deaths', 'recovered']]
Fig5 = Fig5.groupby(['country', 'date']).sum().reset_index()
Fig5.sort_values('date', ascending=True, inplace=True)

## Visualize the data
px.scatter(Fig4, 
           x="confirmed", 
           y="deaths", 
           animation_frame="date", 
           animation_group="country", 
           height = 800,
           size="confirmed", 
           color="deaths", 
           hover_name="country", 
           color_continuous_scale='Reds',
           title = 'Correlation between confirmed cases and death cases from COVID-19 (22-01-2020 to 29-03-2020 )',
           range_color=[0,5000],
           log_x=True,
           text ='country',
           size_max=100, 
           range_x=[100,300000], 
           range_y=[-9000,100000])
#Confirmed COVID-19 count around the world
world1['size'] = world1['confirmed'].pow(0.2)
fig6 = display_scatter_geo(world1, "country",'country names', "confirmed",
                                 "country", "size",
                                 "natural earth",'Confirmed COVID-19 cases of each country(22-01-2020 to 29-03-2020)')
fig6
#Recovery count around the world
world1['size'] = world1['recovered'].pow(0.2)
fig7 = display_scatter_geo(world1, "country",'country names', "recovered",
                                 "country", "size",
                                 "natural earth",'COVID-19 Recovery count of each country(22-01-2020 to 29-03-2020)')
fig7
#Death count around the world
world1['size'] = world1['deaths'].pow(0.2)
fig5 = display_scatter_geo(world1, "country",'country names', "deaths",
                                 "country", "size",
                                 "natural earth",'Fatalities count of each country(22-01-2020 to 29-03-2020)')
fig5
#Confirmed Cases Over Time
total_cases = df2.groupby('date')['date', 'confirmed'].sum().reset_index()
total_cases['date'] = pd.to_datetime(total_cases['date'])

fig6, plt = create_figure(16,10,"Worldwide Confirmed Cases Over Time", 20, True,None,None, None,None,None,None)
sns.pointplot(x=total_cases.date.dt.date, y=total_cases.confirmed, color = 'r')
plt = set_fig_xy_label(plt, 'Dates', 'Total Cases', 15, 15)
plt.xticks(rotation = 90 ,fontsize = 10)
plt.yticks(fontsize = 15)
# displaying data frame of the top 10 countries for confirmed, active and death cases
top = df2[df2['date'] == df2['date'].max()]

# Top 10 countries having the most number of confirmed cases
top_confirmed = top.groupby(by = 'country')['confirmed'].sum().sort_values(ascending = False).head(10).reset_index()

# Top 10 countries having the most number of death cases
top_deaths = top.groupby(by = 'country')['deaths'].sum().sort_values(ascending = False).head(10).reset_index()

# call widget to display data frame 
widget1 = create_widget_Text_w_df("Top 10 confirmed cases", top_confirmed)
widget2 = create_widget_Text_w_df("Top 10 death cases", top_deaths)
    
# create HBox
hbox = widgets.HBox([widget1, widget2])

# render hbox
hbox
fig7, plt = create_figure(28,28,"", 14, True,None,None, None,None,0.5,0.25)

# set the value for each variable
fig_font = 20
xlabel_font = 18
ylabel_font = 18
xtick_font = 16 
ytick_font = 16 
val_font = 16

# confirmed Cases
ax1 = plt.subplot2grid((2,1),(0,0)) #1st diagram 
ax1_1 = display_barplot(ax1, top_confirmed.country, top_confirmed.confirmed, "Top 10 countries with the most confirmed cases", 
                'Total Cases', 'Country', fig_font, xlabel_font, ylabel_font, xtick_font, ytick_font, val_font)

# death cases
ax3 = plt.subplot2grid((2,1),(1,0)) #3rd diagram 
ax3_1 = display_barplot(ax3, top_deaths.country, top_deaths.deaths, "Top 10 countries having most deaths cases", 
                'Total Cases', 'Country', fig_font, xlabel_font, ylabel_font, xtick_font, ytick_font, val_font)
rate = top.groupby(by = 'country')['confirmed','deaths'].sum().reset_index()
rate['death percentage'] =  round(((rate['deaths']) / (rate['confirmed'])) * 100 , 2)
rate.head()
# displaying data frame of the top 10 mortality Rate

# mortality
mortality = rate.groupby(by = 'country')['death percentage'].sum().sort_values(ascending = False).head(10).reset_index()

fig8, plt = create_figure(30,12,"", 14, True,None,None, None,None,0.65,None)

# set the value for each variable
fig_font = 20
xlabel_font = 18
ylabel_font = 18
xtick_font = 17 
ytick_font = 17 
val_font = 17

# mortality Cases
ax1 = plt.subplot2grid((1,2),(0,0)) #1st diagram 
ax1_1 = display_barplot(ax1, mortality.country, mortality['death percentage'], "Top 10 countries having highest mortality rate", 
                'Total Cases', 'Country', fig_font, xlabel_font, ylabel_font, xtick_font, ytick_font, val_font)
# displaying data frame of the 4 countries in comparison
lst_arr = ['Singapore','US','China','Italy','Spain','France']

# initialisation of variable
i_index = 0
select_ctry_df = []

widget_arr = [None] * len(lst_arr) # initialise the widget array to empty to size 4

# loop through the dataframe to filter the required country and display the widget 
for str1 in lst_arr:
    var = df2[df2.country == str1]
    var = var.groupby(by = 'date')['deaths', 'confirmed'].sum().reset_index()    
    widget_arr[i_index] = create_widget_Text_w_df(str1, var.head()) # set to widget based on filter group by data
    var['id'] = var.index # assign the index to a column call id    
    var['country'] = str1
    select_ctry_df.append(var)
    i_index = i_index + 1

select_ctry_df = pd.concat(select_ctry_df) # merge the list of df into one single df

print(select_ctry_df.confirmed.max())

# create Virtual box
left_box = widgets.VBox([widget_arr[0], widget_arr[2]])
right_box = widgets.VBox([widget_arr[1], widget_arr[3]])
    
# create HBox
hbox = widgets.HBox([left_box, right_box])

# render hbox
hbox

fig9, plt = create_figure(16,10,"Comparision of Confirmed Cases Over Time", 20, True,None,None, None,None,None,None)
sns.pointplot(select_ctry_df.id, select_ctry_df.confirmed, hue=select_ctry_df.country, data=select_ctry_df)
plt = set_fig_xy_label(plt, 'No of Days', 'Total Confirmed Cases', 15, 15)
fig10, plt = create_figure(16,10,"Comparision of Fatality Cases Over Time", 20, True,None,None, None,None,None,None)
sns.pointplot(select_ctry_df.id, select_ctry_df.deaths, hue=select_ctry_df.country, data=select_ctry_df)
plt = set_fig_xy_label(plt, 'No of Days', 'Total Fatality Cases', 15, 15)