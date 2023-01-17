# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from IPython.display import Image
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/air-quality-data-in-india/city_day.csv')
df['Date']=pd.to_datetime(df['Date'])
df['day_name']=df['Date'].dt.day_name()
df['Month'] = df['Date'].dt.strftime('%m-%d')
cdict = {'Good':'green',  'Satisfactory': 'aqua', 'Moderate': 'yellow',  
         'Poor':'maroon',   'Very Poor':'red' ,    'Severe': 'black' }
week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
def colour(data,cdict):
    return [cdict[x] for x in data.index]
city_data = df[df['City'] == 'Delhi']
for i in sorted(set(city_data['Date'].dt.year)):
    city_date = pd.DataFrame(city_data['AQI_Bucket'][city_data['Date'].dt.year == i ].value_counts())
    city_date.plot(kind='pie',x='index',y='AQI_Bucket',title='Delhi AQI Year- ' + str(i),colors=colour(city_date,cdict),legend=False,figsize=(5,5))
agent_days=[]
agents = city_data.columns.drop(['City', 'Date','AQI_Bucket', 'day_name','Month'])
for dn in week_days:
    for agent in agents:
        city_agent = city_data[city_data[agent].notna()]
        dn_data = city_agent[agent][city_agent['day_name'] == dn].mean()
        agent_days.append([agent,dn,dn_data])
for agent in agents:
    agent_days =pd.DataFrame(agent_days,columns=['Agents','Days','Past_Avg_5_years'])
    agents_mean = agent_days[['Days','Past_Avg_5_years']][(agent_days['Agents'] == agent)]
    agents_mean.plot(kind='line',x='Days',y='Past_Avg_5_years',legend=False,title= agent +' - Avearge Level(2015-2020)')
Image(url="https://w.ndtvimg.com/sites/3/2019/12/18122322/air_quality_index_standards_CPCB_650.jpg")
def past_years(city):
    last_3yr = [2018,2019,2020]
    city_data = df[df['City'] == city]
    plt.figure(figsize=(16,6))
    for i in last_3yr:
        time = city_data[['Month','AQI']][(city_data['Date'].dt.year == i) & (city_data['Month'].between('03-23','04-15'))].set_index('Month')
        plt.plot(time,linewidth=3, markersize=12,marker='o')
        plt.xlabel('23 Mar - 15 Apr')
        plt.ylabel('AQI Level (<100 is Satisfactory)')
        plt.title(city + ' Air - Quality through Years')
        plt.legend(last_3yr)
past_years('Delhi')
city_15_20 = pd.DataFrame(city_data['AQI_Bucket'].value_counts())
city_15_20.plot(kind='barh',y='AQI_Bucket',color=colour(city_15_20,cdict),legend=False,title='Level of Pollution (2015-2020) in Days')
plt.xlabel('Count(Days)',fontsize=14)