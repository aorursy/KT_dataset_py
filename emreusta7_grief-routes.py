import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

        #import numpy as np

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
data = pd.read_csv("../input/missing-migrants-project/MissingMigrants-Global-2019-03-29T18-36-07.csv")
data.info()
data.head(3)
data.drop('Reported Date', axis=1, inplace=True)

data.drop('Information Source', axis=1, inplace=True)

data.drop('URL', axis=1, inplace=True)

data.drop('UNSD Geographical Grouping', axis=1, inplace=True)
data['Number Dead'].fillna(0, inplace=True)

data['Minimum Estimated Number of Missing'].fillna(0, inplace=True)

data['Total Dead and Missing'].fillna(0, inplace=True)

data['Number of Survivors'].fillna(0, inplace=True)

data['Number of Females'].fillna(0, inplace=True)

data['Number of Males'].fillna(0, inplace=True)

data['Number of Children'].fillna(0, inplace=True)
data['Number Dead'] = data['Number Dead'].astype(int)

data['Minimum Estimated Number of Missing'] = data['Minimum Estimated Number of Missing'].astype(int)

data['Total Dead and Missing'] = data['Total Dead and Missing'].astype(int)

data['Number of Survivors'] = data['Number of Survivors'].astype(int)

data['Number of Females'] = data['Number of Females'].astype(int)

data['Number of Males'] = data['Number of Males'].astype(int)

data['Number of Children'] = data['Number of Children'].astype(int)
data.head(3)
data.describe()
roi = data['Region of Incident'].value_counts()

roi = roi[roi > 100]

print(roi)



roi.plot(kind='barh', color = '#00b8a9')

plt.title('Regions with most incidents')

plt.ylabel('Number of Incidents')

plt.xlabel('Regions')

#plt.setp(lines, color='r', linewidth=2.0)

plt.show()
minus2019 = data[data['Reported Year'] != 2019]

dead = minus2019[minus2019['Number Dead'] > 0]
yearly_dead = dead.groupby('Reported Year')['Number Dead'].sum()

print(yearly_dead)
yearly_dead.plot(color = '#f67280')

plt.title('Loss of Life Through Years')

plt.ylabel('Total Loss of Life')

plt.xlabel('Year')

plt.show()
#I dont know why, but this agg dropped the year column from the df or changed the level of it.

gender = dead.groupby('Reported Year').agg(

        female=('Number of Females', sum),

        male=('Number of Males', sum)

)



print(gender)
#Adding year column

year = [2014,2015,2016,2017,2018]

gender['Year'] = year
plt.bar(gender['Year'], gender['female'], color="#f3e151")

plt.bar(gender['Year'], gender['male'], bottom=gender['female'], color="#6c3376")

plt.legend(['Female','Male'])

plt.title('Total Loss of Life by gender through years')
cause = data.groupby('Cause of Death')['Number Dead'].sum()

print(cause.sort_values(ascending = False))
cause = cause.nlargest(10)

cause.plot(kind='barh', color = '#4a69bb')

plt.title('Top 10 Cause of Death')

plt.ylabel('Cause of Death')

plt.xlabel('Total Loss of Life')

plt.show()