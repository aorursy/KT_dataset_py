# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt   #Data Visualisation 

import seaborn as sbn



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
space_data = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')

space_data.head(2)
space_data.shape
space_data.info()
space_data.isna().sum()
space_data.drop(['Unnamed: 0','Unnamed: 0.1',' Rocket'],axis=1,inplace=True)
space_data.head(2)
space_data['country_of_launch'] = space_data['Location'].apply(lambda x:x.split(',')[-1])

space_data.head(2)
space_data['date'] = pd.to_datetime(space_data['Datum'],errors='coerce')

space_data['year_of_launch'] = space_data['date'].apply(lambda dt: dt.year)
space_data.drop(columns=('Datum'),axis=1,inplace=True)
space_data.head(2)
# DATA ANALYSIS
top_15_companies=space_data['Company Name'].value_counts().head(15)

top_15_companies
plt.figure(figsize=[17,7])

plt.style.use('seaborn-white')

top_15_companies.plot.bar(color=sbn.color_palette('Set1'))

plt.xlabel('Space Agencies',fontsize=20)

plt.ylabel('Launches',fontsize=20)

plt.tick_params(labelsize=13)

plt.title('TOTAL LAUNCHES PER SPACE AGENCY',fontsize=30)

plt.tight_layout()

plt.show()
most_preferred_loc=space_data.Location.value_counts().head(10)[::-1]

most_preferred_loc
most_preferred_loc.plot.barh(color=sbn.color_palette('magma'))

plt.xlabel('No.of Launches',fontsize=20)

plt.ylabel('Launch Sites/Locations',fontsize=20)

plt.tick_params(labelsize=13)

plt.title('MOST PREFERRED LAUNCH LOCATIONS',fontsize=30)

plt.show()
preferred_countries=space_data.country_of_launch.value_counts().head(10)

preferred_countries
plt.figure(figsize=[17,7])

preferred_countries.plot.bar(color=sbn.color_palette('rocket'))

plt.xlabel('Country',fontsize=20)

plt.ylabel('No.of Launches',fontsize=20)

plt.tick_params(labelsize=13)

plt.title('PREFERRED COUNTRIES FOR LAUNCH',fontsize=30)

plt.show()
status = space_data['Status Rocket'].value_counts()

status
lab= ['StatusRetired','StatusActive']

col=['#ff1a1a','#0073e6']
plt.figure(figsize=[7,7])

plt.pie(status,labels=lab,wedgeprops={'edgecolor':'black'},autopct='%1.1f%%',colors=col,explode=[0.0,0.05],shadow=True)

plt.title('STATUS ROCKET',fontsize=30)

plt.show()
plt.figure(figsize=[17,7])

plt.style.use('seaborn-whitegrid')

sbn.countplot('Company Name',data=space_data,hue='Status Rocket',palette='seismic',order=space_data['Company Name'].value_counts().head(15).index)

plt.xlabel('Space Companies',fontsize=20)

plt.ylabel('No.of Launches',fontsize=20)

plt.tick_params(labelsize=13)

plt.title('STATUS OF ROCKETS LAUNCHED',fontsize=30)

plt.tight_layout()

plt.show()
mission_status = space_data['Status Mission'].value_counts()

mission_status
lab = ['Success','Failure','Partial Failure','Prelaunch Failure']
plt.figure(figsize=[8,8])

plt.pie(mission_status,labels=lab,wedgeprops={'edgecolor':'black'},shadow=1,autopct='%1.2f%%',explode=[0.0,0.0,0.0,0.09],colors=sbn.color_palette('Set3'))

plt.title('MISSION STATUS',fontsize=30)

plt.tight_layout()

plt.show()
plt.figure(figsize=[17,7])

plt.hist(space_data['year_of_launch'],edgecolor='#000000',color='brown')

plt.xlabel('Year Of Launch',fontsize=20)

plt.ylabel('No.of Launches',fontsize=20)

plt.title('NO.OF LAUNCHES PER FROM 1956-2020',fontsize=30)

plt.tick_params(labelsize=13)

plt.show()