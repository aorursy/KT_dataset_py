# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn

import matplotlib.pyplot as plt 

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


atlantic_hurricanes = pd.read_csv('/kaggle/input/hurricane-database/atlantic.csv')

atlantic_hurricanes.head



df = pd.DataFrame(atlantic_hurricanes)

df.shape #shows dataframe size - rows, columns

booleans =[]



for date in df['Date']:

    if date > 19500000:

        booleans.append(True)

    else: 

        booleans.append(False)

date_range = pd.Series(booleans)

new_df = df[date_range]



new_df.shape

new_df #filtered by date 1950-2015
#drop unnecessary titles

new_df.drop(['ID', 'Time' ,'Name', 'Minimum Pressure', 'Event'], axis = 1)
#removing Hemisphere tags from coordiantes:

new_df['Longitude'] = new_df['Longitude'].map(lambda x: x.rstrip('W'))

new_df['Latitude'] = new_df['Latitude'].map(lambda x: x.rstrip('N'))

new_df['Latitude'] = new_df['Latitude'].map(lambda x: x.rstrip('S'))

new_df['Longitude'] = new_df['Longitude'].map(lambda x: x.rstrip('E'))



new_df
#converting Latitude and Longitude to floats:

new_df['Latitude'] = new_df['Latitude'].astype(float)

new_df['Longitude'] = new_df['Longitude'].astype(float)



#filtering outside latitudes

lat_filtered_df = new_df[(new_df['Latitude'].astype(float) >= float(9)) & (new_df['Latitude'].astype(float) <= float(26))]



print(lat_filtered_df.shape)



#filtering out longitude by coordinate

#convert longitude to negative

lat_filtered_df['Longitude'] = (lat_filtered_df['Longitude'] * -1)

lat_long_filtered_df = lat_filtered_df[(lat_filtered_df['Longitude'] >= -86) & (lat_filtered_df['Longitude'] <= float(-56))]

print(lat_long_filtered_df.shape)



#from the clearly reduced row numbers, it is clear to see that filter has worked
#Converting all data to strings so can be searched

all_columns = list(lat_long_filtered_df) # Creates list of all column headers

lat_long_filtered_df[all_columns] = lat_long_filtered_df[all_columns].astype(str)



#Converting all '-999' null values to 'NaN' which Python can automatically remove:

lat_long_filtered_df = lat_long_filtered_df.replace('-999', np.nan)



#the year 1967 has '-99' as maximum wind speed values - these must also be changed

lat_long_filtered_df = lat_long_filtered_df.replace('-99', np.nan)

lat_long_filtered_df
#Changing date format:

import datetime as t

#df['DateTime'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')



lat_long_filtered_df['Date'] = pd.to_datetime(lat_long_filtered_df['Date'].astype(str), format = '%Y %m %d')

lat_long_filtered_df



#adding year column



lat_long_filtered_df['Year'] = lat_long_filtered_df['Date'].map(lambda x: x.year)



lat_long_filtered_df



#checking for negative values

negatives =[]

for i in lat_long_filtered_df['Maximum Wind']:

    if float(i) < 0:

        negatives.append(i)

        

print(negatives)
#dropping duplicate storms



lat_long_filtered_df = lat_long_filtered_df.sort_values(by='Maximum Wind', ascending=False)

lat_long_filtered_df = lat_long_filtered_df.drop_duplicates(subset='Name', keep="first")



#resort by year

lat_long_filtered_df = lat_long_filtered_df.sort_values(by ='Year', ascending = True)

lat_long_filtered_df.shape



lat_long_filtered_df

#left with 229 storms over 65 year period
#averages 



#average storms per year



#total number of storms = 229 ; years = 65 



print(229/65)



#average wind speed over 65 years



print(lat_long_filtered_df['Maximum Wind'].sum()/229)
#creating dataframe of number of storms per year



number_of_storms={}





count = lat_long_filtered_df['Year'].value_counts()









count_df = pd.DataFrame(count)



count_df = count_df.reset_index()







count_df = count_df.rename(columns={"index": "Year", "Year": "Count"})



count_df = count_df.sort_values(by = "Year", ascending = True)

count_df

#create averages

lat_long_filtered_df['Maximum Wind'] = lat_long_filtered_df['Maximum Wind'].astype(float)

lat_long_filtered_df.Year = lat_long_filtered_df.Year.astype(int)





grouped_df = lat_long_filtered_df.groupby(['Year'])



described_df = grouped_df.describe()



described_df = described_df.reset_index()





described_df = pd.DataFrame(described_df)



described_df.columns = ['Year', 'Count', "Mean", 'std', 'min', '25%', '50%', '75%', 'Max']



described_df
#adding month column to new dataframe

month_df = lat_long_filtered_df

month_df = month_df.drop(['Year'], axis=1)



month_df['Month'] = month_df['Date'].map(lambda x: x.month)

#group by month



month_df['Maximum Wind'] = month_df['Maximum Wind'].astype(float)

month_df.Month = month_df.Month.astype(int)



grouped_df_month = month_df.groupby(['Month'])



described_df_month = grouped_df_month.describe()



described_df_month = described_df_month.reset_index()





described_df_month= pd.DataFrame(described_df_month)



described_df_month.columns = ['Month', 'Count', "Mean", 'std', 'min', '25%', '50%', '75%', 'Max']



described_df_month
#import colours

import plotly.express as px

from textwrap import wrap



named_colorscales = px.colors.named_colorscales()

print("\n".join(wrap("".join('{:<12}'.format(c) for c in named_colorscales), 96)))
#creating bar chart, of year, count, and average speed 

import plotly.express as px



fig_1 = px.bar(described_df, 

                 x= 'Year',

                 y='Count', 

                 color = 'Mean',

                 color_continuous_scale=px.colors.sequential.OrRd,

                 title = 'Frequency and Average Wind Speed <br>of Large Storms in Caribbean (1950-2015)',

                 labels={'Count':'Number of Large Storms', 'Mean' : 'Average Wind <br> Speed (knots)'}

                )



fig_1.update_layout(title_x=0.5)

fig_1.show()
#creating bar chart, of month, count, and average speed 

import plotly.express as px



fig_2 = px.bar(described_df_month, 

                 x= 'Month',

                 y='Count', 

                 color = 'Mean',

                 color_continuous_scale=px.colors.sequential.OrRd,

                 title = 'Frequency and Average Max. Wind Speed <br> of Large Storms in Caribbean by Month <br> (1950-2015) <br>',

                 text = 'Mean',

                 labels={'Count':'Number of Large Storms', 'Mean' : 'Average Maximum <br> Wind Speed (knots)'}

                )



fig_2.update_layout(

                    title_x=0.5,

                    xaxis = dict(

        tickmode = 'array',

        tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

        ticktext = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                                )

                    )

fig_2.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig_2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

fig_2.show()
import plotly.graph_objects as go



fig = go.Figure(data=go.Scattergeo(

    

    lat = lat_long_filtered_df['Latitude'],

    lon = lat_long_filtered_df['Longitude'],



)

                )





fig.update_geos(

    center=dict(lon=-71, lat= 17),

    lataxis_range=[7, 29], lonaxis_range=[-88, -55]

                )
#test using scatter_geo



import plotly.graph_objects as go

fig4 = go.Figure(go.Densitymapbox(

                                  lat=lat_long_filtered_df.Latitude, 

                                  lon=lat_long_filtered_df.Longitude,

                                  z=lat_long_filtered_df['Maximum Wind'],

                                  #colorscale = True,

                                  radius=30, 

                                  opacity =0.5,

                                  hoverinfo = 'none',

                            

    

                                  

                                 ))

fig4.update_layout( title = "Heatmap of Maximum Wind Speed (knots)", title_x = 0.5, mapbox_style= "stamen-terrain", mapbox_center_lon= -76, mapbox_center_lat = 17)



fig4.show()
import plotly.express as px



fig_num = px.bar(count_df, 

                 x='Year', 

                 y='Count', 

                 title = 'Number of Large Storms in Caribbean by Year (1950-2015)',

                 labels={'Count':'Number of Large Storms'}

                )



fig_num.update_layout(title_x=0.5)

fig_num.show()
# Hurricanes Per year



import plotly.graph_objects as go

import numpy as np



fig5 = go.Figure(data=go.Scatter(

    y = lat_long_filtered_df['Maximum Wind'],

    x = lat_long_filtered_df['Year'],

    mode = 'markers'

))

fig5.show()


fig5 = px.scatter(lat_long_filtered_df, x="Year", y="Maximum Wind", color='Maximum Wind',

                 title="Maximum Wind Speed of Storms (knots) by Year")



fig5.show()