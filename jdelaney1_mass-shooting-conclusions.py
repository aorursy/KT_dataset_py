# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib as mpl

import seaborn as sns

from geopy.geocoders import Nominatim

color = sns.color_palette()



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



ms = pd.read_csv("../input/Mass Shootings Dataset.csv", encoding = "ISO-8859-1", parse_dates=["Date"])

print("Data Dimensions are: ", ms.shape)
ms.head()
ms.columns
ms = ms.dropna()
shootings_date = ms[['S#', 'Date']]



shootings_date.head()
plt.plot(np.sort(shootings_date['Date']), np.sort(shootings_date['S#']))

plt.xlabel('Date', fontsize=12)

plt.ylabel('Shooting number', fontsize=12)

plt.show()
ms['Fatalities'].sum()
ms['Injured'].sum()
ms['Total victims'].sum()
ms_group = ms[['Date', 'Total victims', 'Injured', 'Fatalities']]
ms_groupby = ms_group.groupby('Date')
ms_fatalities_total = ms_groupby.sum()
fatalties_plot = ms_fatalities_total.plot(kind='bar')
ms_fatalities_total
#as you can see abovethere are too many different dates to plot them by year, so i parsed out the year and month

# needed to add dataframe to ['Date']

ms['Date'] = pd.to_datetime(ms['Date'])

ms['year'], ms['month'] = ms['Date'].dt.year, ms['Date'].dt.month

ms.head()
# and now we'll try the bar chart again



ms_group = ms[['year', 'Total victims', 'Injured', 'Fatalities']]

ms_groupby = ms_group.groupby('year')

ms_fatalities_total = ms_groupby.sum()

fatalties_plot = ms_fatalities_total.plot(kind='bar', figsize=(20,10))



ms_attack_year = ms['year'].value_counts()

attacks_plot = ms_attack_year.plot(kind='bar', figsize=(20,10))
cnt_gender = ms['Gender'].value_counts()

sns.barplot(cnt_gender.index, cnt_gender.values)

plt.xlabel('Shooting by Gender', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Attacks by Gender', fontsize=18)



    

plt.show()


cnt_race = ms['Race'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_race.index, cnt_race.values)

plt.xticks(rotation='vertical')

plt.xlabel('Shooters Race', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Attacks by Race', fontsize=18)



    

plt.show()
# breaking up city and state



ms['City'] = ms['Location'].str.rpartition(',')[0]#.str.replace(",", " ")

ms['State'] = ms['Location'].str.rpartition(',')[2]



# extracting the top 20

ms[ms[['City']].apply(lambda x: x[0].isdigit(), axis=1)].head(20)
cnt_city = ms['City'].value_counts()

cnt_city = cnt_city.head(20)

plt.figure(figsize=(12,6))

sns.barplot(cnt_city.index, cnt_city.values)

plt.xticks(rotation='vertical')

plt.xlabel('City', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Attacks by City - top 20', fontsize=18)



    

plt.show()
cnt_state = ms['State'].value_counts()

cnt_state = cnt_state.head(20)

plt.figure(figsize=(12,6))

sns.barplot(cnt_state.index, cnt_state.values)

plt.xticks(rotation='vertical')

plt.xlabel('State', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Attacks by State - top 20', fontsize=18)



    

plt.show()
cnt_mental = ms['Mental Health Issues'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_mental.index, cnt_mental.values)

plt.xlabel('Mental Health Issues', fontsize=12)

plt.ylabel('Number of Attacks', fontsize=12)

plt.title('Mental Health Issues', fontsize=18)



    

plt.show()
from wordcloud import WordCloud

wordcloud = WordCloud(

                         ).generate(str(ms['Summary']))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
wordcloud = WordCloud(

                         ).generate(str(ms['Title']))



plt.figure(figsize=(12,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()