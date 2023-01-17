# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

color = sns.color_palette()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Mass Shootings Dataset Ver 2.csv', encoding="ISO-8859-1")

print(data.shape)

data.head()
# Drop the S# column

data.drop(['S#'], axis=1, inplace=True)



# Convert the date column as appropriate datetime format

data['Date'] = pd.to_datetime(data['Date'])



# Split the date column into year, month and day

data['Year'] = data.Date.dt.year

data['Month'] = data.Date.dt.month

data['Day'] = data.Date.dt.day

data['Weekday'] = data.Date.dt.dayofweek
# Take an overview now

data.info()
# Let's check the number of NaN values before moving on to each column 

data.isnull().sum()
# Let's start with the year first. 

year_counts = data.Year.value_counts()



plt.figure(figsize=(15,10))

sns.barplot(year_counts.index, year_counts.values)

plt.title('Number of mass shooting per year')

plt.xlabel('Year', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(rotation=90)

plt.show()
# What about the months? Which months count for maximum incidents?

month_counts = data.Month.value_counts()



plt.figure(figsize=(10,5))

sns.barplot(month_counts.index, month_counts.values)

months = ('Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')

plt.title('Number of monthly mass shooting')

plt.xlabel('Month', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(12), months)

plt.show()
# What about the days of a month? 

day_counts = data.Day.value_counts()



plt.figure(figsize=(15,10))

sns.barplot(day_counts.index, day_counts.values)

plt.title('Number of mass shooting')

plt.xlabel('Day of the month', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(rotation=90)

plt.show()
# What about the day of the week?

weekday_counts = data.Weekday.value_counts()



plt.figure(figsize=(10,5))

sns.barplot(weekday_counts.index, weekday_counts.values)

days = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')

plt.title('Number of mass shooting')

plt.xlabel('Weekday', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(7), days)

plt.show()
# Fatalities check

print("Maximum number of fatalities in a mass shooting : ", np.max(data['Fatalities']))

print("Minimum number of fatalities in a mass shooting : ", np.min(data['Fatalities']))

print("Average number of fatalities in any mass shooting : ", int(np.mean(data['Fatalities'])))



fat_count = data.Fatalities

plt.figure(figsize=(10,5))

plt.scatter(range(len(fat_count)), np.sort(fat_count.values), alpha=0.7)

plt.title("Fatalities count in mass shooting")

plt.xlabel("Index")

plt.ylabel("Count")

plt.show()
# Moving on to the injured column

print("Maximum number of injured in a mass shooting : ", np.max(data['Injured']))

print("Minimum number of injured in a mass shooting : ", np.min(data['Injured']))

print("Average number of injured in any mass shooting : ", int(np.mean(data['Injured'])))



inj_count = data['Injured']

plt.figure(figsize=(10,5))

plt.scatter(range(len(inj_count)), np.sort(inj_count.values), alpha=0.7)

plt.title("Injured count in mass shooting")

plt.xlabel("Index")

plt.ylabel("Count")

plt.show()
# Let's check the total number of victims in any case

print("Maximum number of victims in a mass shooting : ", np.max(data['Total victims']))

print("Minimum number of victims in a mass shooting : ", np.min(data['Total victims']))

print("Average number of victims in any mass shooting : ", int(np.mean(data['Total victims'])))



victim_count = data['Total victims']

plt.figure(figsize=(10,5))

plt.scatter(range(len(victim_count)), np.sort(victim_count.values), alpha=0.7)

plt.title("Victim count in mass shooting")

plt.xlabel("Index")

plt.ylabel("Count")

plt.show()
# Mental health issues

mental_health_count = data['Mental Health Issues'].value_counts()



plt.figure(figsize=(10,5))

sns.barplot(mental_health_count.index, mental_health_count.values)

plt.title('Mental health')

plt.xlabel('Issue', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(len(mental_health_count.index)), mental_health_count.index)

plt.show()
data['Mental Health Issues'] = data['Mental Health Issues'].apply(lambda x: 'Unknown' if x=='unknown' else x)



mental_health_count = data['Mental Health Issues'].value_counts()



plt.figure(figsize=(10,5))

sns.barplot(mental_health_count.index, mental_health_count.values)

plt.title('Mental health')

plt.xlabel('Issue', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(len(mental_health_count.index)), mental_health_count.index)

plt.show()
race_count = data['Race'].value_counts()



plt.figure(figsize=(15,10))

sns.barplot(race_count.values, race_count.index, orient='h')

plt.xlabel('Count', fontsize=14)

plt.ylabel('Race', fontsize=14)

plt.show()
data['Race'] = data['Race'].apply(lambda x : 'White' if x=='white' else x)

data['Race'] = data['Race'].apply(lambda x : 'Black' if x=='black' else x)



data['Race'] = data['Race'].apply(lambda x : 'White American or European American' 

                                  if x=='White American or European American/Some other Race' else x)



data['Race'] = data['Race'].apply(lambda x : 'Black American or African American' 

                                  if x=='Black American or African American/Unknown' else x)



data['Race'] = data['Race'].apply(lambda x : 'Asian American' if x=='Asian American/Some other race' else x)

data['Race'] = data['Race'].apply(lambda x : 'Unknown' if x=='Two or more races' or x =='unknown' else x)



data['Race'] = data['Race'].apply(lambda x : 'Other' if x=='Some other race' else x)
# Let's check the final results 

race_count = data['Race'].value_counts()



plt.figure(figsize=(15,10))

sns.barplot(race_count.values, race_count.index, orient='h')

plt.xlabel('Count', fontsize=14)

plt.ylabel('Race', fontsize=14)

plt.show()
# Let's move to the Gender column now

gender = data['Gender'].value_counts()



plt.figure(figsize=(10,5))

sns.barplot(gender.index, gender.values)

plt.title('Gender involved in mass shooting')

plt.xlabel('Gender', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(len(gender.index)), gender.index)

plt.show()
data['Gender'] = data['Gender'].apply(lambda x: 'Male' if x=='M' else x)

data['Gender'] = data['Gender'].apply(lambda x: 'Unknown' if x=='M/F' or x=='Male/Female' else x)
# Let's check it once more now

gender = data['Gender'].value_counts()



plt.figure(figsize=(10,5))

sns.barplot(gender.index, gender.values)

plt.title('Gender involved in mass shooting')

plt.xlabel('Gender', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(len(gender.index)), gender.index)

plt.show()
data.Location.value_counts()
# We can split the location column into two different columns: City and State. Let's do that first

data['City'] = data['Location'].str.rpartition(',')[0]

data['State'] = data['Location'].str.rpartition(',')[2]
# Lets' look at the values of the state column

data.State.value_counts()
# Either convert all States to their abbreviations or do the reverse. Because we have only few entries as abbreviations

# we will do the reverse.

"""

1) Replace CA with California

2) Replace NV with Nevada

3) Replace LA with Louisiana

4) Replace PA with Pennsylvania

5) Replace WA with Washington D.C.

"""

data['State'] = data.State.str.strip()

data['State'] = data.State.str.replace('CA', 'California')

data['State'] = data.State.str.replace('NV', 'Nevada')

data['State'] = data.State.str.replace('LA', 'Louisiana')

data['State'] = data.State.str.replace('PA', 'Pennsylvania')

data['State'] = data.State.str.replace('WA', 'Washington D.C.')
# Let's check again

data.State.value_counts()
data.Location.fillna('Unknown', inplace=True)

data.City.fillna('Unknown', inplace=True)

data.State.fillna('Unknown', inplace=True)
# Let's move to the summary column and check the wordcloud

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



wordcloud = WordCloud(background_color='black',

                        stopwords=stopwords,

                        max_words=100,

                        max_font_size=30, 

                        random_state=42).generate(str(data['Summary']))



plt.figure(figsize=(20,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()