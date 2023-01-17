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
df = pd.read_csv("/kaggle/input/nepalcoviddata/nepal_covid(22Jan-18May2020).csv")

district_wise_df = pd.read_csv("/kaggle/input/nepalcoviddata/districtwise_nepalCovidData.csv")
df.head(2)
district_wise_df.head(2)
print(len(district_wise_df))

count_df = district_wise_df[district_wise_df["confirmed"]>=1]

r, c = count_df.shape

print("Total number of affected district: %s districts. There are actually 77 districts in Nepal." %r)
df.plot(color=["blue","red","green"])

plt.show()
# district_wise_df[['confirmed','deaths','recovered']].sum().plot.bar(color=["blue","red","green"])

# plt.show()
group_size = [sum(district_wise_df['confirmed']),

              sum(district_wise_df['recovered']),

              sum(district_wise_df['deaths'])]

group_labels = ['confirmed\n' + str(sum(district_wise_df['confirmed'])),

                'recovered\n' + str(sum(district_wise_df['recovered'])),

                'deaths\n' + str(sum(district_wise_df['deaths']))]

custom_colors = ["blue","green","red"]

plt.figure(figsize = (5,5))

plt.pie(group_size, labels = group_labels, colors = custom_colors)

central_circle = plt.Circle((0,0), 0.5, color = 'white')

fig = plt.gcf()

fig.gca().add_artist(central_circle)

plt.rc('font', size = 12)

plt.title("Nepal's total Confirmed, Recovered and Deaths Cases", fontsize = 20)

plt.show()
df['date']= pd.to_datetime(df['date']) 



months_df = df.set_index('date').groupby(pd.Grouper(freq='M')).sum()

months_df.plot(color=["blue","red","green"])

plt.show()
months_df = df.set_index('date').groupby(pd.Grouper(freq='W')).sum()

months_df.plot(color=["blue","red","green"])

plt.show()
district_wise_df.plot(kind='bar',x='district',y=['confirmed','deaths','recovered'],color=["blue","red","green"])

plt.show()





# district_wise_df.plot(kind='bar',x='district',y='confirmed',color="blue")

# plt.show()

import seaborn as sns

sns.set_style("ticks")

plt.figure(figsize = (15,10))

plt.barh(district_wise_df["district"],district_wise_df["confirmed"].map(int),align = 'center', color = 'lightblue', edgecolor = 'blue')

plt.xlabel('No. of Confirmed cases', fontsize = 18)

plt.ylabel('District', fontsize = 18)

plt.gca().invert_yaxis()

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

plt.title('Total Confirmed Cases Statewise', fontsize = 18 )

for index, value in enumerate(district_wise_df["confirmed"]):

    plt.text(value, index, str(value), fontsize = 12)

plt.show()
# district_wise_df.plot(kind='bar',x='district',y='deaths',color="red")

# plt.show()



sns.set_style("ticks")

plt.figure(figsize = (15,10))

plt.barh(district_wise_df["district"],district_wise_df["deaths"].map(int),align = 'center', color = 'red', edgecolor = 'black')

plt.xlabel('No. of death cases', fontsize = 18)

plt.ylabel('District', fontsize = 18)

plt.gca().invert_yaxis()

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

plt.title('Total death Cases Statewise', fontsize = 18 )

for index, value in enumerate(district_wise_df["deaths"]):

    plt.text(value, index, str(value), fontsize = 12)

plt.show()
# district_wise_df.plot(kind='bar',x='district',y='recovered',color="green")

# plt.show()

plt.figure(figsize = (15,10))

plt.barh(district_wise_df["district"],district_wise_df["recovered"].map(int),align = 'center', color = 'green')

plt.xlabel('No. of recovered cases', fontsize = 18)

plt.ylabel('District', fontsize = 18)

plt.gca().invert_yaxis()

plt.xticks(fontsize = 14)

plt.yticks(fontsize = 14)

plt.title('Total recovered Cases Statewise', fontsize = 18 )

for index, value in enumerate(district_wise_df["recovered"]):

    plt.text(value, index, str(value), fontsize = 12)

plt.show()
top_10_confirmed = district_wise_df.sort_values('confirmed',ascending=False)[:10]

top_10_confirmed
top_10_confirmed.plot(kind='bar',x='district',y=["confirmed"],color="blue")

plt.title('Top 10 affected Districts')

plt.xlabel('District', fontsize=18)

plt.ylabel('count', fontsize=16)

plt.show()
# Comparing with "deaths","recovered"

top_10_confirmed.plot(kind='bar',x='district',y=["confirmed","deaths","recovered"],color=["blue","red","green"])

plt.title('Top 10 affected Districts')

plt.xlabel('District', fontsize=18)

plt.ylabel('count', fontsize=16)

plt.show()
top_10_deaths = district_wise_df.sort_values('deaths',ascending=False)[:10]

top_10_deaths
top_10_deaths.plot(kind='bar',x='district',y=["deaths"],color="red")

plt.title('Top 10 districts with high number of Deaths')

plt.xlabel('District', fontsize=18)

plt.ylabel('count', fontsize=16)

plt.show()
# Comparing with "confirmed","recovered"

top_10_deaths.plot(kind='bar',x='district',y=["confirmed","deaths","recovered"],color=["blue","red","green"])

plt.title('Top 10 districts with high number of Deaths')

plt.xlabel('District', fontsize=18)

plt.ylabel('count', fontsize=16)

plt.show()
top_10_recovered = district_wise_df.sort_values('recovered',ascending=False)[:10]

top_10_recovered
top_10_recovered.plot(kind='bar',x='district',y=["recovered"],color="green")

plt.title('Top 10 Districts where max number of affected Patients recovered')

plt.xlabel('District', fontsize=18)

plt.ylabel('count', fontsize=16)

plt.show()
# Comparing with "confirmed","deaths"

top_10_recovered.plot(kind='bar',x='district',y=["confirmed","deaths","recovered"],color=["blue","red","green"])

plt.title('Top 10 Districts where max number of affected Patients recovered')

plt.xlabel('District', fontsize=18)

plt.ylabel('count', fontsize=16)

plt.show()