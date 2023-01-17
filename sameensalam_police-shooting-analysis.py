# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as datetime

import seaborn as sns

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller

from datetime import datetime

import plotly.graph_objects as go

from wordcloud import WordCloud



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def add_value_labels(ax, spacing=5):

    """Add labels to the end of each bar in a bar chart.



    Arguments:

        ax (matplotlib.axes.Axes): The matplotlib object containing the axes

            of the plot to annotate.

        spacing (int): The distance between the labels and the bars.

    """



    # For each bar: Place a label

    for rect in ax.patches:

        # Get X and Y placement of label from rect.

        y_value = rect.get_height()

        x_value = rect.get_x() + rect.get_width() / 2



        # Number of points between bar and label. Change to your liking.

        space = spacing

        # Vertical alignment for positive values

        va = 'bottom' 



        # If value of bar is negative: Place label below bar

        if y_value < 0:

            # Invert space to place label below

            space *= -1

            # Vertically align label at top

            va = 'top'



        # Use Y value as label and format number with two decimal place

        label = "{:.2f}".format(y_value)



        # Create annotation

        ax.annotate(

            label,                      # Use `label` as label

            (x_value, y_value),         # Place label at end of the bar

            xytext=(0, space),          # Vertically shift label by `space`

            textcoords="offset points", # Interpret `xytext` as offset in points

            ha='center',                # Horizontally center label

            va=va)                      # Vertically align label differently for

                                        # positive and negative values.

def easy_bar_plot(data, variable, title = "", xlab = "", ylab = "Proportion", xtick_rotation = 0): 

    sns.set()

    temp = data[variable].value_counts(dropna = False, normalize = True).to_frame()

    plt.figure(figsize=(10, 6))

    ax = temp[variable].plot(kind='bar',rot = xtick_rotation, color = 'mediumseagreen', edgecolor = 'black')

    ax.set_title(title)

    ax.set_xlabel(xlab)

    ax.set_ylabel(ylab)

    

    add_value_labels(ax)
#Creating a convenient function to plot empirical cumulative distribution plots for continuous

#variables

def ecdf_plot(data, variable, x_lab, display_lines = True):

    

    from statsmodels.distributions.empirical_distribution import ECDF

    

    '''Plot empirical cumulative distribution function of a numerical variable

    and plot mean (red) and median (green) lines

    

    Keyword Arguments: 

    data -- pandas Dataframe 

    variable -- column name (string) **must be a numerical input**

    x_lab -- X axis label (string)

    display_lines -- True or False 

    

    '''

    sns.set()

    ecdf = ECDF(data[variable])

    _= plt.plot(ecdf.x, ecdf.y, marker = ".", linestyle = "none", alpha = 0.2)

    _= plt.xlabel(x_lab)

    _= plt.ylabel("Cumulative Density")

    if (display_lines == True):

        _= plt.vlines(np.nanmean(data[variable]), ymax=1,ymin=0,colors='r')

        _= plt.vlines(np.nanmedian(data[variable]), ymax = 1, ymin = 0, colors = 'g')
full_police_data = pd.read_csv("/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv")
full_police_data.head()
#Getting all metadata of the dataset

full_police_data.info()
#Plotting bar plot of manner_of_death

easy_bar_plot(data= full_police_data, variable= 'manner_of_death', ylab= 'Proportion', title= 'Manner of Death')
#Getting each state's proportion of transactions 

armed_props = full_police_data["armed"].value_counts().to_frame()/full_police_data.shape[0]

print(armed_props)
wordcloud_armed = WordCloud(background_color='white', collocations= False).generate(' '.join(full_police_data.dropna()['armed']))



plt.figure(figsize = (10,10), facecolor = None) 

plt.imshow(wordcloud_armed, interpolation='bilinear') 

plt.axis("off") 

plt.tight_layout(pad = 0) 



plt.show() 
#Creating a list of armed levels that make up at least 1% of the entire dataset for visualization purposes

armed_reason_highenough = armed_props.index[armed_props.armed > 0.01]



#Creating the armed_binned variable that bins all other armed values below 1% of the total into "other"

full_police_data["armed_binned"] = full_police_data['armed'].apply(lambda x : x if x in armed_reason_highenough \

                                                                   else x if pd.notnull(x) == False else 'other')
#Plotting barplot of armed_binned

easy_bar_plot(data=full_police_data, variable= 'armed_binned', title= "Armed Status", xtick_rotation=45)
#Plotting empirical cumulative distribution function of age

ecdf_plot(data= full_police_data, variable= 'age', x_lab= 'Age (years)')
#Plotting barplot of gender

easy_bar_plot(data= full_police_data, variable= 'gender', title= 'Fatal Shootings by Gender', xlab="Gender")
#Plotting barplot of race

easy_bar_plot(data=full_police_data, variable= 'race', xlab="Race", title='Fatal Police Shootings by Race')
#Plotting barplot of signs_of_mental_illness

easy_bar_plot(data=full_police_data, variable= 'signs_of_mental_illness', xlab="Mental Illness Suspected",\

              title = 'Fatal Police Shootings by Appearance of Mental Illness')
#Plotting barplot of threat_level

easy_bar_plot(data=full_police_data, variable= 'threat_level', xlab="Perceived Threat", \

              title= "Fatal Police Shootings by Perceived Threat Level")
#Plotting barplot of flee

easy_bar_plot(data=full_police_data, variable= 'flee', xlab="Flee Status", \

              title= "Fatal Police Shootings by Flee Status")
#Plotting barplot of body_camera

easy_bar_plot(data=full_police_data, variable= 'body_camera', xlab="Body Camera Present", \

             title= "Fatal Police Shootings by Body Camera Presence")
#Getting longitude and latitude data at the city level for visualizations

lat_long_source = pd.read_csv('https://raw.githubusercontent.com/kelvins/US-Cities-Database/master/csv/us_cities.csv')

lat_long_source = lat_long_source.rename(columns={'STATE_CODE':'state', 'CITY':'city'})

lat_long_source.head()
#Left joining full_police_data to lat_long_source on their common state and city columns

merged = pd.merge(left=full_police_data, right=lat_long_source, on=['state', 'city'], how='left')
#Viewing merge results

merged.head()
#Dropping duplicates introduced by multiple geographical coordinates from lat_lon_source dataset

merged = merged.drop_duplicates(subset=['city','name','date','id'])



#Converting the date column to datetime, then creating a year and month column from that

merged['date'] = merged.date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

merged['year'], merged['month'] = merged['date'].dt.year, merged['date'].dt.month
#Getting a dataset where each city (and it's shooting count) is the observation, not a fatal shooting incident

city_as_obs = merged.city.value_counts().to_frame()

city_as_obs.head()

city_as_obs = city_as_obs.reset_index()

city_as_obs.columns = ['city', 'count']



#Left joining city_as_obs with merged, isolating the necessary columns, and displaying the result

city_as_obs_latlon = pd.merge(left=city_as_obs, right=merged, on=['city'], how='left')

city_as_obs_latlon = city_as_obs_latlon[['city', 'count', 'LATITUDE', 'LONGITUDE']]

city_as_obs_latlon = city_as_obs_latlon.drop_duplicates(subset=['city'])

city_as_obs_latlon.info()

city_as_obs_latlon.head()
#Plotting the count of fatal shooting onto a US map according to city

#Size of bubble represents the number, the coloration represents rank order binning (ie purple is for the top 15 highest 

#fatal shooting count among the list of cities)



city_as_obs_latlon['text'] = city_as_obs_latlon['city'] + '<br>Fatal Shootings ' + city_as_obs_latlon['count'].astype(str)

limits = [(0,15),(16,30),(31,50),(51,100),(101,2470)]

colors = ["royalblue","crimson","lightseagreen","orange","lightgrey"]

cities = []

scale = 0.5



fig = go.Figure()



for i in range(len(limits)):

    lim = limits[i]

    df_sub = city_as_obs_latlon[lim[0]:lim[1]]

    fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df_sub['LONGITUDE'],

        lat = df_sub['LATITUDE'],

        text = df_sub['text'],

        marker = dict(

            size = df_sub['count']/scale,

            color = colors[i],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        ),

        name = '{0} - {1}'.format(lim[0],lim[1])))



fig.update_layout(

        title_text = 'Total Fatal Police Shootings in the US from Jan 2015 to June 2020 <br>(Legend represents rank order, not number of police killings)',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    )



fig.show()
#Creating unarmed dataset

unarmed = merged.loc[merged['armed_binned'] == 'unarmed', :]
#Unarmed stacked proportion plot

x = merged.loc[merged['armed_binned'] == "unarmed", :]['race'].value_counts(normalize = True)

x_ = pd.DataFrame([x])

x_.index = ['unarmed']



_= x_.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Unarmed Victims by Race', figsize=(10,8))

x_
#Armed with gun stacked proportion plot

x = merged.loc[merged['armed_binned'] == "gun", :]['race'].value_counts(normalize = True)

x_ = pd.DataFrame([x])

x_.index = ['Armed: Gun']



_= x_.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Victims Armed With Guns by Race', figsize=(10,8))

x_
#threat_level by race stacked proportion plot

black = merged.loc[merged['race'] == 'B', :]['threat_level'].value_counts(normalize = True)

white = merged.loc[merged['race'] == 'W', :]['threat_level'].value_counts(normalize = True)

hispanic = merged.loc[merged['race'] == 'H', :]['threat_level'].value_counts(normalize = True)

asian = merged.loc[merged['race'] == 'A', :]['threat_level'].value_counts(normalize = True)

other = merged.loc[merged['race'] == 'O', :]['threat_level'].value_counts(normalize = True)



x_y = pd.DataFrame([black, white, hispanic, asian, other])

x_y.index = ['Black', 'White', 'Hispanic', 'Asian', 'Other']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Race by Threat Level', figsize=(10,8))

x_y
#Body Camera presence by race stacked proportion plot

black = merged.loc[merged['race'] == 'B', :]['signs_of_mental_illness'].value_counts(normalize = True)

white = merged.loc[merged['race'] == 'W', :]['signs_of_mental_illness'].value_counts(normalize = True)

hispanic = merged.loc[merged['race'] == 'H', :]['signs_of_mental_illness'].value_counts(normalize = True)

asian = merged.loc[merged['race'] == 'A', :]['signs_of_mental_illness'].value_counts(normalize = True)

other = merged.loc[merged['race'] == 'O', :]['signs_of_mental_illness'].value_counts(normalize = True)



x_y = pd.DataFrame([black, white, hispanic, asian, other])

x_y.index = ['Black', 'White', 'Hispanic', 'Asian', 'Other']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Race by Signs of Mental Illness', figsize=(10,8))

x_y
#Gender by race stacked proportion plot

x = merged.loc[merged['gender'] == 'M', :]['race'].value_counts(normalize = True)

y = merged.loc[merged['gender'] == 'F', :]['race'].value_counts(normalize = True)

x_y = pd.DataFrame([x, y])

x_y.index = ['Male', 'Female']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

            title = 'Stacked Proportion Chart of Gender by Race', figsize=(10,8))

x_y
#Body Camera presence by race stacked proportion plot

black = merged.loc[merged['race'] == 'B', :]['body_camera'].value_counts(normalize = True)

white = merged.loc[merged['race'] == 'W', :]['body_camera'].value_counts(normalize = True)

hispanic = merged.loc[merged['race'] == 'H', :]['body_camera'].value_counts(normalize = True)

asian = merged.loc[merged['race'] == 'A', :]['body_camera'].value_counts(normalize = True)

other = merged.loc[merged['race'] == 'O', :]['body_camera'].value_counts(normalize = True)



x_y = pd.DataFrame([black, white, hispanic, asian, other])

x_y.index = ['Black', 'White', 'Hispanic', 'Asian', 'Other']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Race by Body Camera Presence', figsize=(10,8))

x_y
#Body Camera Presence for unarmed victims by race stacked proportion plot

black = unarmed.loc[unarmed['race'] == 'B', :]['body_camera'].value_counts(normalize = True)

white = unarmed.loc[unarmed['race'] == 'W', :]['body_camera'].value_counts(normalize = True)

hispanic = unarmed.loc[unarmed['race'] == 'H', :]['body_camera'].value_counts(normalize = True)

asian = unarmed.loc[unarmed['race'] == 'A', :]['body_camera'].value_counts(normalize = True)

other = unarmed.loc[unarmed['race'] == 'O', :]['body_camera'].value_counts(normalize = True)



x_y = pd.DataFrame([black, white, hispanic, asian, other])

x_y.index = ['Black', 'White', 'Hispanic', 'Asian', 'Other']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Race by Body Camera Presence for Unarmed Victims', figsize=(10,8))

x_y
#Flee status by race stacked proportion plot for unarmed victims

black = unarmed.loc[unarmed['race'] == 'B', :]['flee'].value_counts(normalize = True)

white = unarmed.loc[unarmed['race'] == 'W', :]['flee'].value_counts(normalize = True)

hispanic = unarmed.loc[unarmed['race'] == 'H', :]['flee'].value_counts(normalize = True)

asian = unarmed.loc[unarmed['race'] == 'A', :]['flee'].value_counts(normalize = True)

other = unarmed.loc[unarmed['race'] == 'O', :]['flee'].value_counts(normalize = True)



x_y = pd.DataFrame([black, white, hispanic, asian, other])

x_y.index = ['Black', 'White', 'Hispanic', 'Asian', 'Other']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Flee Status by Race for Unarmed Victims', figsize=(10,8))

x_y
#Armed Status by race stacked proportion plot

black = merged.loc[merged['race'] == 'B', :]['armed_binned'].value_counts(normalize = True)

white = merged.loc[merged['race'] == 'W', :]['armed_binned'].value_counts(normalize = True)

hispanic = merged.loc[merged['race'] == 'H', :]['armed_binned'].value_counts(normalize = True)

asian = merged.loc[merged['race'] == 'A', :]['armed_binned'].value_counts(normalize = True)

other = merged.loc[merged['race'] == 'O', :]['armed_binned'].value_counts(normalize = True)



x_y = pd.DataFrame([black, white, hispanic, asian, other])

x_y.index = ['Black', 'White', 'Hispanic', 'Asian', 'Other']



_= x_y.plot(kind = 'bar', stacked= True, rot = 0, \

                               title = 'Stacked Proportion Chart of Race by Armed Status', figsize=(10,8))

x_y
#Creating additional count column, then isolating necessary columns into new dataframe

merged['count'] = 1 

date_count = merged[["date" ,"count"]]
#Looking at date_count 

date_count.head()
#Get aggregate killings for each of the 12 months in the year

date_count.groupby(date_count['date'].dt.strftime('%B'))['count'].sum().sort_values()
#Get time series of police shootings over month/year

count_monthly = date_count.groupby(date_count['date'].dt.strftime('%B %Y'))['count'].sum().to_frame()

count_monthly.reset_index(inplace=True)

count_monthly['date'] =  pd.to_datetime(count_monthly['date'], format='%B %Y')

count_monthly.head()
#Plot time series of aggregate monthly shootings over time

_= count_monthly[['date','count']].plot('date', figsize=(15,8))

_.set_xlabel("Year");

_.set_ylabel("Count");