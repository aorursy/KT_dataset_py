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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import time



sns.set_color_codes("pastel")

sns.set_style('whitegrid')
state = pd.read_csv(r'/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv')

state.head()
state.isnull().sum()  #checking for null values
state = state.dropna()  #removing null values

state.isnull().sum()
print(state.duplicated().sum())  #checking for duplicates
state['dt'] = pd.to_datetime(state['dt'])

state['dt']
state['Year'] = state['dt'].map(lambda x:x.year)

state['Month'] = state['dt'].map(lambda x:x.month)
state.head()
indian_temp_data = state[state['Country']=='India']

indian_temp_data = indian_temp_data.reset_index(drop=True)

indian_temp_data
states = np.unique(indian_temp_data['State'])

states
min_year = indian_temp_data['Year'].min()

max_year = indian_temp_data['Year'].max()

print(min_year)

print(max_year)
years = range(min_year,max_year+1)

avg_temp = []



for year in years:

    for state in states:

        current_year_data = indian_temp_data[indian_temp_data['Year']==year]

        current_state_data = current_year_data[current_year_data['State']==state]

        avg_temp.append((state,year,current_state_data['AverageTemperature'].mean()))



print(avg_temp[:20])
clean_avg_temp = []

for i in range(len(avg_temp)):

    if not np.isnan(avg_temp[i][2]):   #removing nan values

        clean_avg_temp.append(avg_temp[i])
andaman_data = []

andhra_pradesh_data = []

bihar_data = []

delhi_data = []

gujarat_data = []

jammu_kashmir_data = []

uttar_pradesh_data = []

himachal_data = []

rajasthan_data = []

goa_data = []

mp_data = []

tn_data = []

uttaranchal_data = []



for i in range(len(clean_avg_temp)):

    if clean_avg_temp[i][0] == 'Andaman And Nicobar':

        andaman_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Delhi':

        delhi_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Gujarat':

        gujarat_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Jammu And Kashmir':

        jammu_kashmir_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Uttar Pradesh':

        uttar_pradesh_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Himachal Pradesh':

        himachal_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Rajasthan':

        rajasthan_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Goa':

        goa_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Madhya Pradesh':

        mp_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Tamil Nadu':

        tn_data.append(clean_avg_temp[i])

    elif clean_avg_temp[i][0] == 'Uttaranchal':

        uttaranchal_data.append(clean_avg_temp[i])
years_andaman = []

temp_change_andaman = []

for i in range(len(andaman_data)):

    years_andaman.append(andaman_data[i][1])

    temp_change_andaman.append(andaman_data[i][2])    

    

years_delhi = []

temp_change_delhi = []

for i in range(len(delhi_data)):

    years_delhi.append(delhi_data[i][1])

    temp_change_delhi.append(delhi_data[i][2]) 

    

years_gujarat = []

temp_change_gujarat = []

for i in range(len(gujarat_data)):

    years_gujarat.append(gujarat_data[i][1])

    temp_change_gujarat.append(gujarat_data[i][2]) 

    

years_jammu_kashmir = []

temp_change_jammu_kashmir = []

for i in range(len(jammu_kashmir_data)):

    years_jammu_kashmir.append(jammu_kashmir_data[i][1])

    temp_change_jammu_kashmir.append(jammu_kashmir_data[i][2]) 

    

years_uttar_pradesh = []

temp_change_uttar_pradesh = []

for i in range(len(uttar_pradesh_data)):

    years_uttar_pradesh.append(uttar_pradesh_data[i][1])

    temp_change_uttar_pradesh.append(uttar_pradesh_data[i][2]) 

    

years_himachal = []

temp_change_himachal = []

for i in range(len(himachal_data)):

    years_himachal.append(himachal_data[i][1])

    temp_change_himachal.append(himachal_data[i][2]) 

    

years_rajasthan = []

temp_change_rajasthan = []

for i in range(len(rajasthan_data)):

    years_rajasthan.append(rajasthan_data[i][1])

    temp_change_rajasthan.append(rajasthan_data[i][2]) 

    

years_goa = []

temp_change_goa = []

for i in range(len(goa_data)):

    years_goa.append(goa_data[i][1])

    temp_change_goa.append(goa_data[i][2])

    

years_mp = []

temp_change_mp = []

for i in range(len(mp_data)):

    years_mp.append(mp_data[i][1])

    temp_change_mp.append(mp_data[i][2])

    

years_tn = []

temp_change_tn = []

for i in range(len(tn_data)):

    years_tn.append(tn_data[i][1])

    temp_change_tn.append(tn_data[i][2])

    

years_uttaranchal = []

temp_change_uttaranchal = []

for i in range(len(uttaranchal_data)):

    years_uttaranchal.append(uttaranchal_data[i][1])

    temp_change_uttaranchal.append(uttaranchal_data[i][2])
fig,ax = plt.subplots(figsize=(20,14))

plt.plot(years_andaman,temp_change_andaman,label='Andaman Temperature Change')

plt.plot(years_delhi,temp_change_delhi,label='Delhi Temperature Change')

plt.plot(years_gujarat,temp_change_gujarat,label='Gujarat Temperature Change')

plt.plot(years_jammu_kashmir,temp_change_jammu_kashmir,label='Jammu Kashmir Temperature Change')

plt.plot(years_uttar_pradesh,temp_change_uttar_pradesh,label='Uttar Pradesh Temperature Change')

plt.plot(years_himachal,temp_change_himachal,label='Himachal Temperature Change')

plt.plot(years_rajasthan,temp_change_rajasthan,label='Rajasthan Temperature Change')

plt.plot(years_goa,temp_change_goa,label='Goa Temperature Change')

plt.plot(years_mp,temp_change_mp,label='Madhya Pradesh Temperature Change')

plt.plot(years_tn,temp_change_tn,label='Tamil Nadu Temperature Change')

plt.plot(years_uttaranchal,temp_change_uttaranchal,label='Uttaranchal Temperature Change')

plt.title("Analysis of temperature of different states of India over the years")

plt.xlabel("Years")

plt.ylabel("Average temperature")

plt.legend(loc='center left',frameon=True,bbox_to_anchor=(1, 0.5))

plt.show()