# load packages

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline
# load data

police_data = pd.read_csv('../input/clean_data.csv')



police_data.head(4)
# check the data dimensions

police_data.shape
# check if there is any missing data

police_data.info()
death_sum = police_data.canine.value_counts()



fig = plt.figure(figsize=(6, 4))

death_sum.plot(kind = 'bar', color = 'skyblue')



plt.title("Summary of Death")

plt.ylabel("Total Cases")

plt.xticks(death_sum.index, ['Police', 'Canine'], rotation=45)

plt.show()
# filter canine death

death_cause = police_data[police_data['canine']==False].cause_short.value_counts()



fig = plt.figure(figsize=(6, 8))

death_cause.plot(kind='barh', color = 'skyblue')



plt.title('Summary of Police Death by Cause')

plt.xlabel('Total Cases')

plt.show()
death_year = police_data[police_data['canine']==False].year.value_counts().sort_index()



fig = plt.figure(figsize=(8, 6))



plt.plot(death_year, color='brown')



plt.title('Death by Year')

plt.ylabel("Total Cases")

plt.xticks(np.arange(death_year.index[0]-1, death_year.index[-1] + 15, 30) , rotation=45)

plt.xlabel("Year")

plt.show()
death_state = police_data[police_data['canine']==False].state.value_counts()



fig = plt.figure(figsize=(8, 12))

death_state.plot(kind='barh', color = 'skyblue')



plt.title('Summary of Police Death by State')

plt.xlabel('Total Cases')

plt.show()
death_911 =police_data[(police_data['cause_short']=='9/11 related illness') & \

(police_data['canine']==False)].year.value_counts().sort_index()



fig = plt.figure(figsize=(8, 6))

death_911.plot(kind='bar', color = 'skyblue')

plt.ylabel("Total Cases")

plt.title('9/11 Related Death by Year')

plt.show()