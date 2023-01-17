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
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans

%matplotlib inline

plt.style.use('bmh')
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df.head()
states = df['Province/State']
country = df['Country/Region']
places = zip(states, country)
places_list = [str(s) + '_' + str(c) for s,c in places]
df.insert(2, 'place(state_Country)', places_list)
helper_df = df.drop(["Province/State", "Country/Region"], axis=1)
helper_df = helper_df.set_index(list(helper_df.columns)[0:3])
helper_df.head()
stacked_df = helper_df.stack(dropna = False)
stacked_df = stacked_df.reset_index()
stacked_df = stacked_df.rename(columns={'level_3':'date', 0 : 'confirmed'})
stacked_df
plt.figure()
g = sns.FacetGrid(data=stacked_df[:(len(helper_df.columns)*24)],col='place(state_Country)',col_wrap=4, height=2, aspect=1.3)
sns.set(font_scale=0.7)
sca = plt.scatter
m = g.map(sca,'date','confirmed')
g.set_xticklabels(rotation=30, horizontalalignment='right')
# places where the new cases of covid-19 is 5 or less in last 15 days.
under_ctrl_df = helper_df[(helper_df['3/22/20'] - helper_df['3/7/20'] <= 5) & (helper_df['3/5/20']!=0)]
under_ctrl_df = under_ctrl_df[list(under_ctrl_df.columns)[46:]]
under_ctrl_df[20:40]
# finding the dates when total cases became more than three in the places provided in the data
first_few_df = pd.DataFrame(columns=['place(state_Country)','Lat','Long','date','confirmed'])
for i in range(len(helper_df)):
    place_name = helper_df.index[i][0]
    temp_df = stacked_df[(stacked_df['place(state_Country)'] == place_name) & (stacked_df['confirmed'] >= 3)].iloc[:1]
    first_few_df = first_few_df.append(temp_df, ignore_index = True)
#first_few_df.head(10)
date_nums = np.array([])
# assigning the dates a number
for i in range(len(first_few_df)):
    pos = list(stacked_df['date']).index(first_few_df['date'][i])
    date_nums = np.append(date_nums,pos)
# dividing the dates into 4 clusters
kmeans = KMeans(n_clusters = 4, random_state=0).fit(date_nums.reshape(-1,1))
print(kmeans.labels_)
print(kmeans.cluster_centers_)
# plotting the distribution of dates (of numerical vslues assigned to them) when total cases became more than three
# in different places.
plt.figure(figsize=(5,4))
sns.distplot(date_nums, color='g', bins=100, hist_kws={'alpha':0.7})
#first_few_df = first_few_df.drop(['kmeans_label'], axis=1)
first_few_df.insert(5, 'kmeans_label', kmeans.labels_)
# places where the outbreak happened around day 4.01(label 1),47(label 0), 38(label 3), 55(label 2). 
# (see kmeans cluster centers)
# The color represents the day when number of cases became more than 3 in that place.
fig = px.scatter_geo(first_few_df, lat='Lat', lon='Long', color = 'kmeans_label',
                     hover_name='place(state_Country)', projection='natural earth')
fig.show()
# color according to confirmed cases as on March 22
fig = px.scatter_geo(df, lat='Lat', lon='Long', size = '3/31/20', color = 'Country/Region',
                     hover_name='place(state_Country)', projection='natural earth')
fig.show()
#Thank you for visiting :)
