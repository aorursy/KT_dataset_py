# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Census = '/kaggle/input/2018-population/2018 Population.csv'

df_USA_2018 = pd.read_csv(Census)

df_USA_2018.head()
import matplotlib.pyplot as plt

# Data to plot

labels = 'White', 'African American', 'Asian', 'Others'

sizes = [77.8, 14.1, 6.6, 1.4]

colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','brown']

explode = (0, 0.1, 0, 0)  # explode 1st slice



# Plot



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=140)

ax1.set_title("USA Population % 2018")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
import matplotlib.pyplot as plt; plt.rcdefaults()





deaths = ('African American', 'White', 'Asian', 'American Indian')

y_pos = np.arange(len(deaths))

performance = [774.4,652.3,494.4,363.5]



plt.barh(y_pos, performance, align='center', alpha=1)

plt.yticks(y_pos, deaths)

plt.title('2018 Age Adjusted Death Rate Per 100,000')



plt.show()
#Black Population by Region 2018

Region = '/kaggle/input/blackpopulationbyregion/Black Population by Region.csv'

df_Black_region = pd.read_csv(Region)

df_Black_region.head()


# Data to plot

labels = 'Northeast', 'Midwest', 'South', 'West'

sizes = [17.22, 17.13, 55.53, 10.12]

colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','brown']

explode = (0, 0, 0, 0)  # explode 1st slice



# Plot



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=140)

ax1.set_title("USA Black Population Spread By Region 2018")

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Cause = '/kaggle/input/causeofdeath/cause of death.csv'

df_2018_Death = pd.read_csv(Cause)

df_2018_Death.head()


plt.barh(df_2018_Death["Cause of Death"], df_2018_Death["White %"], color="b")

plt.barh(df_2018_Death["Cause of Death"], df_2018_Death["Black %"], left= df_2018_Death["White %"],color="y")

plt.xlabel('% of cause of death')  

plt.ylabel('Cause of Death')

plt.legend(["White", "Black"])



plt.show()

CovidDeaths = '/kaggle/input/covid-deaths-all-races/Covid Deaths all races.csv'

df_Covid_Allraces = pd.read_csv(CovidDeaths)

df_Covid_Allraces.head()
df_Covid_Allraces.plot.barh(stacked=False,x='Race')

plt.legend(bbox_to_anchor=(1.80, 1))
Black_Death = '/kaggle/input/2018-covid-blackdeaths/2018_Covid_blackdeaths.csv'

df_2018_Covid_Black= pd.read_csv(Black_Death)

df_2018_Covid_Black.head()
df_2018_Covid_Black.plot.bar(stacked=False,x='State')

plt.legend(bbox_to_anchor=(1.80, 1))
White_Death = '/kaggle/input/2018-covid-whitedeaths/White COVID Deaths to 2018 Deaths.csv'

df_2018_Covid_White= pd.read_csv(White_Death)

df_2018_Covid_White.head()
df_2018_Covid_White.plot.bar(stacked=False,x='State')

plt.legend(bbox_to_anchor=(1.80, 1))

Population_Covid = '/kaggle/input/covid-disproportinate-blackdeath-states/Disproportionate_Population_Death.csv'

df_States_Covid= pd.read_csv(Population_Covid)

df_States_Covid.head()
df_States_Covid.plot.barh(stacked=False,x='State')

plt.legend(bbox_to_anchor=(1.80, 1))