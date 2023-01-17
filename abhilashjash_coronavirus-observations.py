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

%matplotlib inline

plt.style.use('ggplot')

plt.figure(figsize=(20,10))
df=pd.read_csv('/kaggle/input/who-covid-19-latest-22nd-august-2020/WHO COVID-19 global data.csv')

df.head()
df.columns
# Total number of countries who's data is been recorded.

len(df[' Country'].value_counts())
# Grouping by country names

df_grouped = df.groupby(' Country')

df_grouped.head()
df['Date_reported'] = pd.to_datetime(df['Date_reported'])

print(df['Date_reported'].min())

print(df['Date_reported'].max())

print("Days of Recorded Data = ",df['Date_reported'].max() - df['Date_reported'].min())
df
India = df[df[' Country'] == 'India']

India
Brazil = df[df[' Country'] == 'Brazil']

Brazil
USA = df[df[' Country'] == 'United States of America']

USA
plt.figure(figsize=(20,10))

plt.plot(USA['Date_reported'],USA[' Cumulative_cases'], label ='United States of America')

plt.plot(Brazil['Date_reported'], Brazil[' Cumulative_cases'], label = 'Brazil')

plt.plot(India['Date_reported'], India[' Cumulative_cases'], label = 'India')

plt.xlabel('Date',size=20)

plt.ylabel('Cumulative Cases',size=20)

plt.rc('xtick', labelsize=15)

plt.rc('ytick', labelsize=15)

plt.legend(loc = 'best')

params = {'legend.fontsize': 20,

          'legend.handlelength': 2}

plt.rcParams.update(params)
plt.figure(figsize=(20,10))

plt.plot(USA['Date_reported'], USA[' New_cases'], label ='United States of America', color = 'g')

plt.plot(Brazil['Date_reported'], Brazil[' New_cases'], label = 'Brazil', color = 'b')

plt.plot(India['Date_reported'], India[' New_cases'], label = 'India')

plt.xlabel('Date',size=20)

plt.ylabel('New Cases',size=20)

plt.rc('xtick', labelsize=15)

plt.rc('ytick', labelsize=15)

plt.legend(loc = 'best')

params = {'legend.fontsize': 20,

          'legend.handlelength': 2}

plt.rcParams.update(params)
# Countries with 300000 plus cases

df2 = df[df[' Cumulative_cases'] >300000]

df2
df2[' Country'].unique()
len(df[df[' Cumulative_cases'] >300000][' Country'].unique())
df[df[' Cumulative_cases'] >350000][' Country'].unique()
len(df[df[' Cumulative_cases'] >350000][' Country'].unique())
Russia = df[df[' Country'] == 'Russian Federation']

Russia
RSA = df[df[' Country'] == 'South Africa']

RSA
plt.figure(figsize=(20,10))

plt.plot(USA['Date_reported'], USA[' Cumulative_cases'], label ='United States of America',linewidth=3)

plt.plot(Brazil['Date_reported'], Brazil[' Cumulative_cases'], label = 'Brazil',linewidth=3)

plt.plot(India['Date_reported'], India[' Cumulative_cases'], label = 'India',linewidth=3)

plt.plot(Russia['Date_reported'], Russia[' Cumulative_cases'], label = 'Russian Federation',linewidth=3)

plt.plot(RSA['Date_reported'], RSA[' Cumulative_cases'], label = 'South Africa',linewidth=3)

plt.xlabel('Date',size=20)

plt.ylabel('Cumulative Cases in millions',size=20)

plt.rc('xtick', labelsize=15)

plt.rc('ytick', labelsize=15)

plt.legend(loc = 'best')

params = {'legend.fontsize': 20,

          'legend.handlelength': 2}

plt.rcParams.update(params)

plt.yticks(np.arange(0,5000000,500000))

#ax.ticklabel_format(style='plain',useOffset=False)
Germany = df[df[' Country'] == 'Germany']

Germany
UK = df[df[' Country'] == 'The United Kingdom']

UK
Italy = df[df[' Country'] == 'Italy']

Italy
France = df[df[' Country'] == 'France']

France
Belgium = df[df[' Country'] == 'Belgium']

Belgium
plt.figure(figsize=(30,20))

plt.plot(UK['Date_reported'], UK[' Cumulative_cases'], label ='The United Kingdom',linewidth=5)

plt.plot(Germany['Date_reported'], Germany[' Cumulative_cases'], label = 'Germany',linewidth=5)

plt.plot(France['Date_reported'], France[' Cumulative_cases'], label = 'France',linewidth=5)

plt.plot(Italy['Date_reported'], Italy[' Cumulative_cases'], label = 'Italy',linewidth=5)

plt.plot(Belgium['Date_reported'], Belgium[' Cumulative_cases'], label = 'Belgium',linewidth=5)

plt.xlabel('Date',size=20)

plt.ylabel('Cumulative Cases',size=20)

plt.rc('xtick', labelsize=25)

plt.rc('ytick', labelsize=25)

plt.legend(loc = 'best')

params = {'legend.fontsize': 25,

          'legend.handlelength': 2}

plt.rcParams.update(params)
Mexico = df[df[' Country'] == 'Mexico']

Mexico
Peru = df[df[' Country'] == 'Peru']

Peru
Chile = df[df[' Country'] == 'Chile']

Chile
Spain = df[df[' Country'] == 'Spain']

Spain
Columbia = df[df[' Country'] == 'Columbia']

Columbia
plt.figure(figsize=(30,20))

plt.plot(UK['Date_reported'], UK[' Cumulative_deaths'], label ='The United Kingdom',linewidth=5)

plt.plot(Germany['Date_reported'], Germany[' Cumulative_deaths'], label = 'Germany',linewidth=5)

plt.plot(France['Date_reported'], France[' Cumulative_deaths'], label = 'France',linewidth=5)

plt.plot(Italy['Date_reported'], Italy[' Cumulative_deaths'], label = 'Italy',linewidth=5)

plt.plot(Belgium['Date_reported'], Belgium[' Cumulative_deaths'], label = 'Belgium',linewidth=5)

#plt.plot(USA['Date_reported'], USA[' Cumulative_deaths'], label ='United States of America',linewidth=5)

#plt.plot(Brazil['Date_reported'], Brazil[' Cumulative_deaths'], label = 'Brazil',linewidth=5)

#plt.plot(India['Date_reported'], India[' Cumulative_deaths'], label = 'India',linewidth=5)

plt.plot(Russia['Date_reported'], Russia[' Cumulative_deaths'], label = 'Russian Federation',linewidth=5)

plt.plot(Mexico['Date_reported'], Mexico[' Cumulative_deaths'], label = 'Mexico',linewidth=5)

plt.plot(Peru['Date_reported'], Peru[' Cumulative_deaths'], label = 'Peru',linewidth=5, linestyle ='--')

plt.plot(Chile['Date_reported'], Chile[' Cumulative_deaths'], label = 'Chile',linewidth=5, linestyle ='--')

plt.plot(Spain['Date_reported'], Spain[' Cumulative_deaths'], label = 'Spain',linewidth=5, linestyle ='--')

#plt.plot(Columbia['Date_reported'], Columbia[' Cumulative_deaths'], label = 'Columbia',linewidth=5)

plt.plot(RSA['Date_reported'], RSA[' Cumulative_deaths'], label = 'South Africa',linewidth=5, linestyle ='--')

plt.xlabel('Date',size=20)

plt.ylabel('Cumulative_deaths',size=20)

plt.rc('xtick', labelsize=25)

plt.rc('ytick', labelsize=25)

plt.legend(loc = 'best')

params = {'legend.fontsize': 25,

          'legend.handlelength': 2}

plt.rcParams.update(params)