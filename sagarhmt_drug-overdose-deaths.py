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

from wordcloud import WordCloud

import datetime as dt

%matplotlib inline
df = pd.read_csv('/kaggle/input/drug-overdose-deaths/drug_deaths.csv')

df.head()
df.info()
columns_to_drop = ['Unnamed: 0', 'ID', 'DateType', 'ResidenceCounty', 'ResidenceState', 'COD',

                   'OtherSignifican', 'DeathCityGeo', 'ResidenceCityGeo', 'InjuryCityGeo',

                  'DeathCounty', 'LocationifOther', 'Other', 'LocationifOther']

df.drop(columns = columns_to_drop, inplace = True)
def autolabel(graph):

    for patch in graph.patches:

        height = patch.get_height()

        plt.annotate(height.astype('int64'), xy = (patch.get_x() + patch.get_width()/2, height), xytext = (0, 5),

                    textcoords = 'offset points', ha = 'center', fontsize = 13)
year = pd.to_datetime(df['Date']).dt.year.value_counts()

plt.figure(figsize = (10, 5))

with plt.style.context('fivethirtyeight'):

    graph1 = sns.barplot(x = year.index.astype('int64'), y = year.values.astype('int64'), 

                         palette=sns.cubehelix_palette(8))

plt.tight_layout()

plt.ylabel('Deaths')

autolabel(graph1)

plt.show()
labels = df['Race'].value_counts().nlargest(4).index

val = df['Race'].value_counts().nlargest(4).values

val = [val[i] for i in [0, 2, 3, 1]]

plt.pie(val, labels = labels, autopct = '%1.1f%%', shadow = True, pctdistance=0.5, startangle=90, radius = 1.3,

       textprops={'fontweight':'bold'})

age = df['Age']

with plt.style.context('fivethirtyeight'):

    plt.hist(age, bins = range(0, 100, 10), edgecolor = 'black', color = 'tab:purple')

    plt.xticks(range(0, 100, 10))

    plt.xlabel('Age')

    plt.ylabel('Deaths')
residence_city = df['ResidenceCity'].copy().dropna()

residence_city_cloud = ' '.join(city for city in residence_city)

wc = WordCloud(width=2500, height=1500).generate(residence_city_cloud)

plt.figure(figsize = (10, 8))



plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.margins(x = 0, y = 0)

plt.show()
drugs = df.loc[::, 'Heroin':'AnyOpioid']

drugs['Fentanyl'] = drugs['Fentanyl'].replace(['1-A', '1 POPS', '1 (PTCH)'], '1')

drugs['AnyOpioid'] = drugs['AnyOpioid'].replace({'N':'0'})

drugs['Morphine_NotHeroin'] = drugs['Morphine_NotHeroin'].replace(['STOLE MEDS', 'PCP NEG', '1ES', 'NO RX BUT STRAWS'], '1')





drugs[['Fentanyl_Analogue', 'Morphine_NotHeroin', 'AnyOpioid', 'Fentanyl']] = drugs[['Fentanyl_Analogue', 'Morphine_NotHeroin', 'AnyOpioid', 'Fentanyl']].astype('int64')

drug = drugs.sum().sort_values(ascending = False).index

frequency = drugs.sum().sort_values(ascending = False).values

plt.figure(figsize = (10, 6))

plt.tight_layout()

with plt.style.context('fivethirtyeight'):

    s = sns.barplot(x = frequency, y = drug, palette=sns.color_palette("Reds_r", 16))

    for patch in s.patches:

        plt.annotate(patch.get_width().astype('int64'), xy = (patch.get_width(), patch.get_y() + patch.get_height()/2),

                    xytext = (1, 0), textcoords = 'offset points', va = 'center', fontsize = 13)

plt.xlabel('Number of times drugs involved in Deaths', fontsize = 13) 
male = df['Sex'].value_counts().values[0]

female = df['Sex'].value_counts().values[1]

plt.pie([male, female], labels = ['Male', 'Female'], autopct = '%1.1f%%', shadow = True, pctdistance=0.5,

        startangle=90, radius = 1.5, wedgeprops={'edgecolor':'white'}, 

        textprops={'fontweight':'bold', 'fontsize':16})

plt.show()
df['MannerofDeath'] = df['MannerofDeath'].replace(['accident', 'ACCIDENT'], 'Accident')

df['MannerofDeath'].value_counts()
df['InjuryPlace'].value_counts().nlargest(5)
description_of_injury = df['DescriptionofInjury'].copy()

DOI = description_of_injury.replace({'substance abuse':'Substance Abuse', 'SUBSTANCE ABUSE':'Substance Abuse',

                              'Substance abuse':'Substance Abuse', 'drug use': 'Drug Use',

                              'Drug use': 'Drug Use', 'Drug abuse':'Drug Abuse', 'Used Heroin':"Heroin Use",

                                     'Heroin use':'Heroin Use', 'Abuse Substance':'Substance Abuse'}).dropna()

DOI_cloud = ' '.join(description for description in DOI)

wc = WordCloud(width=2500, height=1500, scale = 0.5, background_color = 'lavender').generate(DOI_cloud)

plt.figure(figsize = (7, 5))



plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.margins(x = 0, y = 0)

plt.show()
death_city = df['DeathCity'].copy().dropna()

death_city_cloud = ' '.join(city for city in death_city)

wc = WordCloud(width=2500, height=1500, background_color = 'red').generate(death_city_cloud)

plt.figure(figsize = (9, 7))



plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.margins(x = 0, y = 0)

plt.show()