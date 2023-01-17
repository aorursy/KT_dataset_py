# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_kor_corona_case = pd.read_csv('/kaggle/input/coronavirusdataset/Case.csv')

df_kor_corona_case.head()
df_kor_corona_case.shape
df_kor_corona_case.columns
df_kor_corona_case.isna().sum()/df_kor_corona_case.shape[0]*100
df_province_confirmed = df_kor_corona_case.groupby(['province'])['confirmed'].sum().sort_values(ascending=False).reset_index()

df_province_confirmed.head()
import seaborn as sns

sns.catplot(x='confirmed',y= 'province', data = df_province_confirmed ,kind='bar',height=5, aspect = 2)
df_city_confirmed = df_kor_corona_case.groupby(['city'])['confirmed'].sum().sort_values(ascending=False).reset_index()

sns.catplot(x='confirmed',y= 'city', data = df_city_confirmed ,kind='bar',height=5, aspect = 2)
df_kor_corona_case.loc[df_kor_corona_case['province']=='Daegu'].head(1)
df_infection_confirmed = df_kor_corona_case.groupby(['infection_case'])[['confirmed']].sum().sort_values(by=['confirmed'],ascending=False).reset_index()
sns.catplot(y='infection_case',x= 'confirmed', data = df_infection_confirmed ,kind='bar',height=5, aspect = 2)
df_kor_corona_case.loc[(df_kor_corona_case['infection_case']=='Shincheonji Church') 

                       & (df_kor_corona_case['province'] != 'Daegu')][['province','city','infection_case','confirmed']].sort_values(by=['confirmed'],ascending=False).reset_index()
temp_df_kor_corona_case= df_kor_corona_case.loc[(df_kor_corona_case['infection_case']=='Shincheonji Church') 

                       & (df_kor_corona_case['province'] != 'Daegu')][['province','city','infection_case','confirmed']].sort_values(by=['confirmed'],ascending=False).reset_index().merge(df_province_confirmed, on='province', how='left')
temp_df_kor_corona_case.rename(columns={"confirmed_x": "confirmed_from_shincheonji", "confirmed_y":"confirmed_total"}, inplace=True)
temp_df_kor_corona_case.columns
temp_df_kor_corona_case['percentage_from_shincheonji'] = temp_df_kor_corona_case['confirmed_from_shincheonji']/temp_df_kor_corona_case['confirmed_total']*100

temp_df_kor_corona_case
sns.catplot(y='confirmed_from_shincheonji',x= 'province', data = temp_df_kor_corona_case ,kind='bar',height=5, aspect = 2).set_xticklabels(rotation=90)
sns.catplot(y='percentage_from_shincheonji',x= 'province', data = temp_df_kor_corona_case ,kind='bar',height=5, aspect = 2).set_xticklabels(rotation=90)
df_kor_corona_time = pd.read_csv('/kaggle/input/coronavirusdataset/Time.csv')

df_kor_corona_time.head()
df_kor_corona_time.isna().sum()/df_kor_corona_time.shape[0]*100
total_test = df_kor_corona_time['test'].sum()

test_duration = df_kor_corona_time.shape[0]

first_day = df_kor_corona_time['date'].head(1).values[0]

last_day = df_kor_corona_time['date'].tail(1).values[0]

print("This dataset show us there are {0} test during {1} days since {2} until {3} period".format(total_test,test_duration,first_day,last_day))
negative = df_kor_corona_time['negative'].sum()

confirmed = df_kor_corona_time['confirmed'].sum()

total_test = df_kor_corona_time['test'].sum()

unknown = (df_kor_corona_time['test'].sum()-df_kor_corona_time['confirmed'].sum()-df_kor_corona_time['negative'].sum())

unknown_percentage = round((total_test - negative - confirmed)/total_test*100)

print("From {0} test has been taken by Korean government, they got {1} negative case and {2} confirmed case. Unfortunately there is unknown result from the test. There are {3} case that the results is unknwown. It takes {4} % from total test.".format(total_test,negative,confirmed,unknown,unknown_percentage))
import matplotlib.pyplot as plt



negative_percentage = round(df_kor_corona_time['negative'].sum()/df_kor_corona_time['test'].sum()*100)

confirmed_percentage = round(df_kor_corona_time['confirmed'].sum()/df_kor_corona_time['test'].sum()*100)



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Confirmed COVID-19', 'Negative COVID-19', 'Unknown Result'

sizes = [confirmed_percentage, negative_percentage,unknown_percentage ]

explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

my_colors = ['Coral','turquoise','plum']



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90, colors=my_colors)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

import matplotlib.pyplot as plt



released_percentage = df_kor_corona_time['released'].sum()/df_kor_corona_time['confirmed'].sum()

deceased_percentage = df_kor_corona_time['deceased'].sum()/df_kor_corona_time['confirmed'].sum()

isolated_percentage = (df_kor_corona_time['confirmed'].sum()-df_kor_corona_time['released'].sum()-df_kor_corona_time['deceased'].sum())/df_kor_corona_time['confirmed'].sum()



# Pie chart, where the slices will be ordered and plotted counter-clockwise:

labels = 'Released', 'Deceased', 'Isolated'

sizes = [released_percentage, deceased_percentage,isolated_percentage ]

explode = (0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

my_colors = ['Coral','turquoise','plum']



fig1, ax1 = plt.subplots()

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90, colors=my_colors)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()

percentage_tes_population = round(df_kor_corona_time['test'].sum()/(51.47*1000000)*100)

print('Until 22nd March 2020, Korean goverment success did COVID-19 test to {}% of Korean population.'.format(percentage_tes_population))
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np



plt.style.use('seaborn')



#size

plt.figure(figsize=(17,7))



#data

x_labels = list(df_kor_corona_time['date'].values)

y_case_test = df_kor_corona_time['test'].values

y_case_confirmed = df_kor_corona_time['confirmed'].values



#plot colors

plt.plot(y_case_test, color='blue', label='Tested')

plt.plot(y_case_confirmed, color='red', label='Confirmed')



#title

plt.title('The amount of tested people increase significantly')



#labels

plt.xlabel('Date')

plt.ylabel('sum')

plt.xticks(np.arange(len(x_labels)),x_labels, rotation=90)



# #legend

# blue_patch = mpatches.Patch(color='blue', label='Tested')

# red_patch = mpatches.Patch(color='red', label='Confirmed')



#grid

plt.grid(True)



plt.legend()

plt.show()
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import numpy as np



plt.style.use('seaborn')



#size

plt.figure(figsize=(17,7))





#data

x_labels = list(df_kor_corona_time['date'].values)

y_case_deceased = df_kor_corona_time['deceased'].values

y_case_confirmed = df_kor_corona_time['confirmed'].values

y_case_released = df_kor_corona_time['released'].values



#plot colors

plt.plot(y_case_deceased, color='red',label='Deceased')

plt.plot(y_case_confirmed, color='blue', label='Confirmed/Postive')

plt.plot(y_case_released, color='green', label='Released')



#title

plt.title('Confirm COVID-19 case distribution')



#labels

plt.xlabel('Date')

plt.ylabel('sum')

plt.xticks(np.arange(len(x_labels)),x_labels, rotation=90)



#legend

# red_patch = mpatches.Patch(color='red', label='Deceased')

# blue_patch = mpatches.Patch(color='blue', label='Confirmed/Postive')

# green_patch = mpatches.Patch(color='green', label='Released')



#grid

plt.grid(True)



plt.legend()

plt.show()
