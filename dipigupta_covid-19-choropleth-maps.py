# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import pycountry
#importing the covid 19 data till 9th May 

#confirmed cases,recovered cases and death cases from John Hokinson's University Dataset
confirmed_links="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

recovered_links="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

death_links="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
confirmed_cases = pd.read_csv(confirmed_links)

recovered_cases=pd.read_csv(recovered_links)

death_cases=pd.read_csv(death_links)
confirmed_cases.head()
confirmed_cases.drop(["Province/State","Lat","Long"],axis=1,inplace=True) #dropping the columns which have no relation to our plots
confirmed_df=confirmed_cases.groupby("Country/Region",as_index=False).sum() 

confirmed_df1=confirmed_df.set_index("Country/Region")

confirmed_df2=confirmed_df1.transpose() 

confirmed_df2.columns.rename(" ",inplace=True)

confirmed=confirmed_df2.unstack()  #converting panel data to time series data

confirmed.shape
confirmed.head()
confirmed_data=confirmed.to_frame()

confirmed_data.reset_index(inplace=True)

confirmed_data.columns=["Country","Date","ConfirmedCases"]

confirmed_data.sort_values(by="Date",inplace=True)

confirmed_data.head()
confirmed_data.reset_index(drop=True,inplace=True) # reset the index
countries=confirmed_data["Country"].unique().tolist() 
d_country_code = {}

for country in countries:

    try:

        country_data = pycountry.countries.search_fuzzy(country)

        # country_data is a list of objects of class pycountry.db.Country

        # The first item  ie at index 0 of list is best fit

        # object of class Country have an alpha_3 attribute

        country_code = country_data[0].alpha_3

        d_country_code.update({country: country_code})

    except:

        print('could not add ISO 3 code for ->', country)

        # If could not find country, make ISO code ' '

        d_country_code.update({country: ' '})
#Manually entering the code for which the ISO codes were not found
d_country_code["Taiwan"]="TWN"

d_country_code["Korea, South"]="KOR"

d_country_code["Burma"]="BUR"

d_country_code["Congo (Brazzaville)"]="COG"

d_country_code[" Congo (Kinshasa)"]="COD"

#for the rest ISO codes were not listed
for k, v in d_country_code.items():

    confirmed_data.loc[(confirmed_data.Country == k), 'iso_alpha'] = v #matching the key value pair for the countries and replacing them with ISO codes
import plotly.express as px
scl = [[0.0, '#ffffff'],[0.2, '#ff9999'],[0.4, '#FF7F50'], \

       [0.6, '#ff1a1a'],[0.8, '#cc0000'],[1.0, '#FFA500']] 
fig = px.choropleth(confirmed_data, locations="iso_alpha",

                    color="ConfirmedCases", 

                    hover_name="Country", # column to add to hover information

                      color_continuous_scale= scl,

                    animation_frame= "Date")

fig.show() #Day by Day the visualization of the covid cases WorldWide
recovered_cases.drop(["Province/State","Lat","Long"],axis=1,inplace=True)
recovered_df1=recovered_cases.groupby("Country/Region",as_index=False).sum()

recovered_df2=recovered_df1.set_index("Country/Region")

recovered_df3=recovered_df2.transpose()

recovered_df3.columns.rename(" ",inplace=True)

recovered=recovered_df3.unstack()

recovered.shape
recovered_data=recovered.to_frame()

recovered_data.reset_index(inplace=True)

recovered_data.columns=["Country","Date","RecoveredCases"]

recovered_data.sort_values(by="Date",inplace=True)
for k, v in d_country_code.items():

    recovered_data.loc[(recovered_data.Country == k), 'iso_alpha'] = v
scl = [[0.0, '#F0FFF0'],[0.2, '#98FB98'],[0.4, '#9ACD32'], \

       [0.6, '#228B22'],[0.8, '#3CB371'],[1.0, '#006400']]  #color coding for Recovered map
fig = px.choropleth(recovered_data, locations="iso_alpha",

                    color="RecoveredCases", 

                    hover_name="Country", # column to add to hover information

                      color_continuous_scale= scl,

                    animation_frame= "Date")

fig.show()
death_cases.drop(["Province/State","Lat","Long"],axis=1,inplace=True)
death_df=death_cases.groupby("Country/Region",as_index=False).sum()

death_df1=death_df.set_index("Country/Region")

death_df2=death_df1.transpose()

death_df2.columns.rename(" ",inplace=True)

death=death_df2.unstack()

death.shape
death_data=death.to_frame()

death_data.reset_index(inplace=True)

death_data.columns=["Country","Date","DeathCases"]

death_data.sort_values(by="Date",inplace=True)
death_data=death_data.reset_index(drop=True)
for k, v in d_country_code.items():

    death_data.loc[(death_data.Country == k), 'iso_alpha'] = v

scl = [[0.0, '#ffffff'],[0.2, '#ffe6e6'],[0.4, '#ff8080'], \

       [0.6, '#ff4d4d'],[0.8, '#ff3333'],[1.0, '#e60000']] 
fig = px.choropleth(death_data, locations="iso_alpha",

                    color="DeathCases", 

                    hover_name="Country", # column to add to hover information

                      color_continuous_scale= scl,

                    animation_frame= "Date")

fig.show()