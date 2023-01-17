# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 50000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.head()
df.shape
dict(df['Country/Region'].value_counts())

"""cases grouped according to date"""
## Mercy's comments:
# You are converting "ObservationDate" to a datetime field and reassigning it to a different column.
# You should either create a totally new column or reassign it back to the "ObservationDate" column.

df["Province/State"] = pd.to_datetime(df["ObservationDate"])
region_wise= df.groupby(['ObservationDate']).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
print(region_wise)
df.tail(80)
## Mercy's comments:
# If you check the sums recorded for Nigeria, you'll see that with the way you are aggregating the values, 
# we have a lot more cases down below than we actually have. 
# The reason behind this is that you are summing all the daily recorded cases together. 
# A better alternative would be to use the .max() function. 
# When I use the .max() function here, I'm getting the maximum values for these 3 columns because 
# the values are incremented daily, so the maximum holds the most recent values for every country. 

"""cases grouped according to Country/region"""
grouping = df.groupby(['Country/Region']).agg({"Confirmed":'max',"Recovered":'max',"Deaths":'max'}).reset_index()
print(grouping)


all =len(df["Country/Region"].unique())
print('The total number of countries with disease spread is ' + str(all))

## Mercy's comments:
# Your numbers here are incorrect because of how the data is currently structured in the dataset, 
# which is daily counts of the number of cases for each country. 
# To get the real sums, you'll need to get the actual counts of confirmed, 
# recovered and deaths in each of the countries and then sum that up.
# i.e from the `grouping` variable you created 
all_1 = sum(df["Confirmed"])
print("\nThe Current total number of confirmed cases around the world is  ",str(all_1))


all_2 = sum(df["Recovered"])
print("\nThe total number of recorverd cases around the world is ",str(all_2))

all_3 = sum(df["Deaths"])
print("\nThe total number of death cases around the word is ", str(all_3))

maximum_number =max(df['Confirmed'])
print('\nThe maximum_number currently gotten in a coutry is',str(maximum_number) )

## Mercy's comments: Here is how I would have gotten the actual sums

all_grouping_1 = sum(grouping["Confirmed"])
print("\n___________________________ Accurate Results __________________________")
print("\nThe Current total number of actual confirmed cases around the world is  ",str(all_grouping_1))


all_grouping_2 = sum(grouping["Recovered"])
print("\nThe total number of actual recorverd cases around the world is ",str(all_grouping_2))

all_grouping_3 = sum(grouping["Deaths"])
print("\nThe total number of actual death cases around the word is ", str(all_grouping_3))

maximum_number_grouping =max(grouping['Confirmed'])

print('\nThe actual maximum_number currently gotten in a coutry is',str(maximum_number_grouping) )
grouping["Confirmed"]
plt.figure(figsize = (80, 20 ), dpi = 50)
sns.barplot(x=grouping["Country/Region"], y=grouping["Confirmed"])
plt.xlabel('COUNTRY/REGION', size=60)
plt.ylabel('CONFIRMED CASES', size=60)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (80, 20 ), dpi = 50)
sns.barplot(x=df["ObservationDate"], y=df["Confirmed"])
plt.xlabel('OBSERVATION DATES', size=60)
plt.ylabel('CONFIRMED CASES', size=60)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize = (80, 20 ), dpi = 50)
sns.barplot(x=df["ObservationDate"], y=df["Deaths"])
plt.xlabel('OBSERVATION DATES', size=60)
plt.ylabel('DEATH CASES', size=60)
plt.xticks(rotation=90)
plt.show()
