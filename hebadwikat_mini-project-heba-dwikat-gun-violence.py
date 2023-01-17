# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading libraries 



from datetime import date, timedelta, datetime

import seaborn as sns #apparently, this create more attractive graphs

import pandas_datareader.data as web

%matplotlib inline

import matplotlib.pyplot as plt

import chart_studio.plotly as py

import plotly.graph_objs as go

from datetime import date, timedelta, datetime
#Loading libraries 

 

import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime

import seaborn as sns #apparently, this create more attractive graphs

import pandas_datareader.data as web

%matplotlib inline

import matplotlib.pyplot as plt

from datetime import date, timedelta, datetime 







import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
#first i would like to check the columns that this file have and check which column i want to keep

US_immigration= pd.read_csv('/kaggle/input/gunviolence/MPI-Data-Hub_USImmigFlow_since1820_2018.csv')





# Drop rows with any empty cells

US_immigration=US_immigration.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

#Checking columns and rows number

print ('This  dataset has {0} years of data and {1} features'.format(US_immigration.shape[0],US_immigration.shape[1]))



#the immigrants numbers is seprated by commas , I want to remove that to access it as numerical number

US_immigration['Number of Legal Permanent Residents'] = US_immigration['Number of Legal Permanent Residents'].str.replace(',', '')



US_immigration['immigrants'] = pd.to_numeric(US_immigration['Number of Legal Permanent Residents'])

US_immigration.head()

#Checking this 

US_immigration.describe()
US_immigration.plot(x='Year', y='immigrants' , figsize=(12,7), color=["blue"])

plt.legend(prop={"size":16})

plt.xlabel('Year', fontsize=14)

plt.ylabel('Count of Immigrants', fontsize=14)

plt.title("Number of Legal Permanent Residents",fontsize = 18);

#first i would like to check the columns that this file have and check which column i want to keep

gun_violence = pd.read_csv('../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')

#Checking columns and rows number

print ('This  dataset has {0} incidients and {1} features'.format(gun_violence.shape[0],gun_violence.shape[1]))

gun_violence.head()
#separate year from date

gun_violence['Year'] = pd.DatetimeIndex(gun_violence['date']).year



#aggregate data to group by year and sum the number of victums

gun_violence_agg = gun_violence.groupby('Year')['n_killed'].sum().to_frame()



#pick the correct data range

immigViolence = US_immigration[-6:].drop('Number of Legal Permanent Residents', axis=1)



#set year to be index for joining later

immigViolence = immigViolence.set_index('Year')



#join 2 datasets

JoinedSet = gun_violence_agg.join(immigViolence)



#find the correlation between immigration and number of people killed

JoinedSet.corr()
# Draw a jointplot between Number of Person Killed Vs Injured in all incidences

sns.jointplot("n_killed",

              "immigrants",

              JoinedSet,

              kind='reg',      # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }, optional

               )
mass_shooting = pd.read_csv('/kaggle/input/gunviolence/Mass Shooting 2018 - US states.csv')

mass_shooting = mass_shooting.drop('Operations' , axis = 1)

#renaming the # killed and # injured

mass_shooting = mass_shooting.rename(columns = { "# Killed":"killed","# Injured":"injured"})

mass_shooting.head()
killed = mass_shooting.groupby('State', as_index=False).agg({'killed':'sum', 'injured':'sum'})

state_to_code = {'District of Columbia' : 'dc','Mississippi': 'MS', 'Oklahoma': 'OK', 'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI', 'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 'Nevada': 'NV', 'Maine': 'ME'}

killed['code'] = killed['State'].apply(lambda x : state_to_code[x])

killed.sort_values(by=['killed'], ascending=False) #sorting by the highest number of people killed
# scl = [[0.0, 'rgb(242,240,245)'],[0.2, 'rgb(218,220,245)'],[0.4, 'rgb(188,100,220)'],\

         #   [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']] #creating a scale for the map



# data = [ dict(

      #  type='choropleth',

      #  colorscale = scl,

     #   autocolorscale = False,

     #   locations = killed['code'],

      #  z = killed['killed'],

    #    locationmode = 'USA-states',

    #    text = killed['State'],

    #    marker = dict(

       #     line = dict (

    #            color = 'rgb(255,255,255)',

    #            width = 1

       #     ) ),

     #   colorbar = dict(

     #       title = "# of people killed by state ")

     #   ) ]



#layout = dict(

    #    title = 'State wise number of Mass Shooting kills (2018- 2020)',

     #   geo = dict(

         #   scope='usa',

       #     projection=dict( type='albers usa' ),

          #  showlakes = True,

         #   lakecolor = 'rgb(255, 255, 255)'),

      #       )

#fig = dict( data=data, layout=layout )

#iplot(fig)
#first i would like to check the columns that this file have and check which column i want to keep

GV_2012 = pd.read_csv('../input/gunviolence/full_data.csv')

#Checking columns and rows number / dropping any unnecassary columns

GV_2012 = GV_2012.drop(columns = ['Unnamed: 0','hispanic'])

GV_2012.head()
GV_2012['age'].mean()

#The mean age for gun related incidents is almost 44
agg_age = GV_2012.groupby('age')['age'].count()

agg_age.plot(color = 'red' , figsize=(8,8))

plt.legend(prop={"size":16})

plt.xlabel('Suspect Age', fontsize=14)

plt.ylabel('Incident #', fontsize=14)

plt.title("Incidents based on age",fontsize = 18);
GV_2012.info()
ageMean = GV_2012.groupby('intent', as_index=False).agg({'age':'mean'})

ageMean.head()
temp = GV_2012["race"].value_counts().head(30)

#temp.iplot(kind='bar', xTitle = 'race', yTitle = "# of incidents", title = 'number of Gun Violence based on race')

temp.plot.bar(figsize = (20,8) , color = "orange" ,fontsize = 20)

plt.legend(prop={"size":20})

plt.xlabel('race', fontsize=14)

plt.ylabel('# of incidents', fontsize=14)

plt.title("number of Gun Violence based on race",fontsize = 18);
temp = GV_2012["intent"].value_counts().head()

#temp.iplot(kind='bar', xTitle = 'Intent', yTitle = "# of incidents", title = 'Gun Violence based on Intent')

temp.plot.bar(figsize = (20,8) , color = "yellow" , fontsize = 20)

plt.legend(prop={"size":16})

plt.xlabel('Intent', fontsize=14)

plt.ylabel('# of incidents', fontsize=14)

plt.title("Gun Violence based on Intent",fontsize = 18);