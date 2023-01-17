# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(style='darkgrid')





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/us-accidents/US_Accidents_June20.csv')
df.head()
df.columns
import matplotlib.pyplot as plt

#%matplotlib inline
accidents = df[['Start_Time','End_Time','State','City', 'County', 'Zipcode','Severity','Weather_Condition','Amenity', 'Bump', 'Crossing',

       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',

       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop','Sunrise_Sunset']]


#x = accidents['State'].value_counts()

plt.figure(figsize=(20,12))

states = sns.countplot(y = 'State', data = accidents, order = accidents['State'].value_counts().index, palette='Reds_d')

plt.ylabel("State", labelpad=10, fontsize=15, weight = 'bold')

plt.xlabel('Number of Traffic Accidents', labelpad=10,fontsize=15, weight='bold')

plt.title('US Traffic Accidents by State (2016-2020)',fontsize=20, weight='bold')
accidents['Start_Time'] = pd.to_datetime(accidents['Start_Time'])

accidents['End_Time'] = pd.to_datetime(accidents['End_Time'])



accidentsCA = accidents[accidents["State"] == 'CA']

accidentsCA['City'].value_counts().head(15)

top15CA = pd.DataFrame(accidentsCA['City'].value_counts().head(15))

plt.figure(figsize=(20,12))

sns.barplot(top15CA['City'],top15CA.index,palette="Reds_d")

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel("Number of Traffic Accidents", labelpad = 10, fontsize=15,weight='bold')

plt.ylabel("City", labelpad = 12,fontsize=15, weight='bold')

plt.title("Top 15 California Cities for Traffic Accidents", fontsize=20,weight='bold')
top_weather_CA = pd.DataFrame(accidentsCA['Weather_Condition'].value_counts().head(15))



plt.figure(figsize=(20,12))

sns.barplot(top_weather_CA['Weather_Condition'],top_weather_CA.index,palette="YlOrRd")

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel("Number of Traffic Accidents", labelpad = 10, fontsize=15,weight='bold')

plt.ylabel("Weather Condition", labelpad = 12,fontsize=15, weight='bold')

plt.title("Top Weather Conditions for California Traffic Accidents", fontsize=20,weight='bold')
surround = ['Amenity', 'Bump', 'Crossing',

       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',

       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

count = {}



for i in range(len(surround)):

    try:

        acc_count = accidentsCA[surround[i]].value_counts()[1]

        count.setdefault(surround[i], acc_count)

    except(KeyError):

        count.setdefault(surround[i],0)
surroundings = pd.DataFrame.from_dict(count, orient = 'index', columns = ['Accidents'])

surroundings = surroundings.sort_values('Accidents', ascending = False)

surroundings
plt.figure(figsize=(20,12))

sns.barplot(surroundings['Accidents'],surroundings.index, palette = 'Blues_r')

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel("Number of Traffic Accidents", labelpad = 10, fontsize=15,weight='bold')

plt.ylabel("Nearby Road Features", labelpad = 12,fontsize=15, weight='bold')

plt.title("Common Road Features and Traffic Accidents in California", fontsize=20,weight='bold')
accidentsCA['Sunrise_Sunset'].value_counts()
plt.figure(figsize = (12,8))

daynight = plt.pie(accidentsCA['Sunrise_Sunset'].value_counts(),autopct='%1.1f%%',shadow = True, explode = (0,0.1), colors = ['tomato','royalblue'], textprops=dict(color="w",weight = 'bold'))

plt.legend(['Day','Night'], loc="best",prop={'size': 15})

plt.title('Traffic Accidents Percentages by Time',fontsize=20,weight='bold')