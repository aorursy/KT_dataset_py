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
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



df = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv', low_memory=False, parse_dates=['Datum'])

df.head()
df.info()
# Remove the following columns: Unnamed: 0, Unnamed: 0.1, Rocket

df.drop(['Unnamed: 0','Unnamed: 0.1',' Rocket'],axis=1,inplace=True)

space_df = pd.DataFrame(df)



# Add Country column from Location column

space_df['Country'] = space_df['Location'].apply(lambda x:x.split()[-1])

# Yearly rocket missions

datetimes=pd.to_datetime(space_df['Datum'], utc=True)

space_df['LaunchYear'] = datetimes.dt.year

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(20,8))

fig = sns.countplot(x="LaunchYear", data=space_df)

fig.set(title='Yearly Rocket Missions since 1957',

       xlabel = 'Launch Year',

       ylabel = 'Number of Rocket Missions');

plt.xticks(rotation='vertical');

fig.axhline(space_df['LaunchYear'].value_counts().mean(), ls='--');
# Rocket launches status

plt.figure(figsize=(10, 6))

fig = sns.countplot(x="Status Mission", data=space_df)

fig.set(title='Total Successful/Failed Launch Missions',

       xlabel = 'Mission Status',

       ylabel = 'Counts');
# Status of the rockets

plt.figure(figsize=(10, 6))

fig = sns.countplot(x="Status Rocket", data=space_df)

fig.set(title='Active and Retired Rockets',

       xlabel = 'Status of Rockets',

       ylabel = 'Counts');
fig = space_df['Country'].value_counts().head(10).plot(kind='bar', x='Country', y='index', figsize=(14,4), color='salmon');

fig.set(title='Rocket Missions by Country (Top 10)',

       xlabel = 'Country',

       ylabel = 'Number of Missions');
# Rocket Launches by Location (Top 20)



fig = space_df['Location'].value_counts().head(20).plot(kind='barh', x='index', y='Location', figsize=(14,8), color='salmon');

fig.set(title='Top 20 Space Launch Facilities',

       xlabel = 'Number of Missions',

       ylabel = 'Launch Location');

fig.invert_yaxis()
# Rocket Missions

x = pd.crosstab(space_df['Country'], space_df['Status Mission'])

x = x.sort_values(by='Success',ascending=False)

fig = x.head(7).plot(kind='bar', figsize=(10, 8), color=['salmon','lightblue','orange','lightgreen'])

fig.set(title='Mission Status by Country',

       xlabel = 'Rocket Counts',

       ylabel = 'Country');
# Active and Retired Rockets by Country

x = pd.crosstab(space_df['Country'], space_df['Status Rocket'])

x = x.sort_values(by='StatusActive',ascending=False)

plt.style.use('seaborn-darkgrid')

fig = x.head(7).plot(kind='barh', figsize=(10, 5), color=['lightblue','salmon'])

fig.set(title='Rocket Status by Country',

       xlabel = 'Rocket Counts',

       ylabel = 'Country');

fig.invert_yaxis()
# Rocket launches by companies

fig = space_df['Company Name'].value_counts().head(20).plot(kind='barh', x='index', y='Company', figsize=(14,8), color='salmon');

fig.set(title='Space Rockets by Company/Manufacturer',

       xlabel = 'Number of Rockets',

       ylabel = 'Company Name/Manufacturer');

fig.invert_yaxis()