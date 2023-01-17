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
covid = pd.read_csv("/kaggle/input/WHO-COVID-19-global-data.csv")
covid.head(10)
from datetime import datetime
covid['day'] = covid['day'].apply(lambda x : datetime.strptime(x,"%Y-%m-%d").strftime('%m/%d/%Y'))
covid.head()
Total_cases = covid[['Country Name','Cumulative Confirmed']].loc[covid['day']=='05/21/2020']
Total_cases.reset_index(drop=True)
Total_cases.sort_values(by=['Cumulative Confirmed'],ascending=False,inplace=True)
Total_cases.head(10)
Total_deaths = covid[['Country Name','Cumulative Deaths']].loc[covid['day']=='05/21/2020']
Total_deaths.reset_index(drop=True)
Total_deaths.sort_values(by=['Cumulative Deaths'],ascending=False,inplace=True)
Total_deaths.head(10)
import seaborn as sb

import plotly_express as px

import matplotlib.pyplot as plt
countries =[]

Total_cases['Country Name'].head(10).apply(lambda x : countries.append(x))
countries
Top_country_cases = pd.DataFrame(columns=['day','Country Name','Confirmed','Cumulative Confirmed'])

for x in countries:

    Top_country_cases = pd.concat([Top_country_cases,covid[['day','Country Name','Confirmed','Cumulative Confirmed']].loc[covid['Country Name'] == x]])
Top_country_cases.reset_index(drop=True)
Top_country_cases.sort_values(by=['day'],inplace=True)
Top_country_cases.head(10)

Top_country_cases = Top_country_cases.astype({'Cumulative Confirmed':'int32','Confirmed':'int32'})
plt.figure(figsize = (30,10))

plt.xticks(rotation = 90)

sb.lineplot(x='day',y='Cumulative Confirmed',hue='Country Name',data=Top_country_cases)
px.pie(Top_country_cases, values='Confirmed', names='Country Name', title='Top 10 Countries in Covid cases ')
plt.figure(figsize = (30,10))

plt.xticks(rotation = 90)

sb.barplot(x='day',y='Cumulative Confirmed',hue='Country Name',data=Top_country_cases)