# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import pycountry

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data

earthquake_data = pd.read_csv('/kaggle/input/global-significant-earthquake-database-from-2150bc/Worldwide-Earthquake-database.csv',index_col=0)

earthquake_data.head() #Preview data
earthquake_data.info()
earthquake_data.describe()
earthquake_data.isnull().sum()
earthquake_data.shape #Total number of row and columns 


data_year= earthquake_data.groupby("YEAR")["FOCAL_DEPTH"].max()



data_year.sort_values(ascending=False).head(10).plot(kind="barh",title="Which year has maximum depthed earthquake?",color='coral', figsize=(10,6))

plt.xlabel("Focal depth")

from_1900to2000=earthquake_data[earthquake_data['YEAR']>=1900] #filter data 



fig = plt.figure(figsize=(16,8)) 

country = from_1900to2000.groupby("COUNTRY")["YEAR"].count()

country.sort_values(ascending=False).head(20).plot(kind="bar",color='tan', title="Which country has faced more number of earthquake from 1900 to 2020?")

plt.ylabel("Total number of earthquake")
fig = plt.figure(figsize=(16,8)) 

colors_list = ['skyblue', 'yellowgreen']

explode_list = [0,0.05]

earthquake_data['T']= np.where(earthquake_data['FLAG_TSUNAMI']=='Yes', 1, 0)

eq=earthquake_data['T'].value_counts()

eq.plot(kind='pie',

        figsize=(15, 6),

        autopct='%1.1f%%', 

        startangle=90,    

        shadow=True,       

        colors=colors_list,

        pctdistance=0.5,# add custom colors

        explode=explode_list, 

        labels=["No Tsunami due to earthquak","Tsunami due to earthquake"]

        )

plt.title("Tsunami due to earthquake") 

plt.legend(labels=["No","Yes"], loc='upper left')
is_tsunami=from_1900to2000[from_1900to2000['FLAG_TSUNAMI']=='Yes']

fig = plt.figure(figsize=(16,8)) 

country = is_tsunami.groupby("COUNTRY")["YEAR"].count()

country.sort_values(ascending=False).head(20).plot(kind="bar",color='limegreen', title="Which country has faced more number of Tsunami from 1900 to 2020?")

plt.ylabel("Total number of Tsunami")
total_death=earthquake_data.groupby('COUNTRY')['DEATHS'].sum().reset_index(name ='D_COUNT')

countries = {}

for country in pycountry.countries:

    countries[country.name.upper()] = country.alpha_3



total_death['code'] = [countries.get(country, np.NaN) for country in total_death['COUNTRY']]



print(total_death[total_death['code'].isnull()])

code_dics={'AZORES (PORTUGAL)':'PRT','BOLIVIA':'BOL','BOSNIA-HERZEGOVINA':'BIH',

       'IRAN':'IRN','KERMADEC ISLANDS (NEW ZEALAND)':'NZL','MACEDONIA':'MKD',

       'MYANMAR (BURMA)':'MMR','NORTH KOREA':'PRK','RUSSIA':'RUS','SOUTH KOREA':'KOR',

       'SYRIA':'SYR','TAIWAN':'TWN','UK':'GBR','USA':'USA','VENEZUELA':'VEN', 'TANZANIA':'TZA', 'VIETNAM':'VNM',

        'CZECH REPUBLIC':'CZE'}

print(total_death.columns)

for key,value in code_dics.items():

    total_death.loc[total_death['COUNTRY'].eq(key), 'code'] = value
total_death.dropna(inplace=True)

import plotly.express as px

import geopandas as gpd

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))



for_plotting = world.merge(total_death,left_on = 'iso_a3', right_on = 'code')



ax = for_plotting.dropna().plot(column='D_COUNT', cmap =    

                                'YlGnBu', figsize=(15,9),   

                                 scheme='quantiles', legend =  

                                  True);

ax.set_title("Earthquake fatalities")

plt.show()