import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from mpl_toolkits.basemap import Basemap



%matplotlib inline
voice_call_sep = pd.read_csv('../input/voice-call-quality-experience-for-november-2019/MyCall_Data_September_2019.csv')

voice_call_sep.head()
voice_call_oct = pd.read_csv('../input/voice-call-quality-experience-for-november-2019/MyCall_Data_October_2019.csv')

voice_call_oct.head()
voice_call_nov = pd.read_csv('../input/voice-call-quality-experience-for-november-2019/MyCall_Data_November_2019.csv')

voice_call_nov.head()
# Shapes of the data across the three months



print(voice_call_sep.shape)

print(voice_call_oct.shape)

print(voice_call_nov.shape)



total_data = voice_call_sep.shape[0] + voice_call_oct.shape[0] + voice_call_nov.shape[0]





# Lets also print the % of data available across all months



print('\nPercentage-wise distribution in the three months: \n\nSept: {}%, \nOct: {}%, \nNov:{}%'.format(round(100* voice_call_sep.shape[0]/total_data, 2), 

                                         round(100* voice_call_oct.shape[0]/total_data, 2),

                                         round(100* voice_call_nov.shape[0]/total_data), 2))
print('September data: ', voice_call_sep.isnull().sum())

print('\n\nOctober data: ', voice_call_oct.isnull().sum())

print('\n\nNovemeber data: ', voice_call_nov.isnull().sum())
# RATING count across months



figure = plt.figure(figsize=(20,5))

plt.subplot(1,3,1)

sns.countplot(voice_call_sep.Rating)

plt.title('Call Rating in September')



plt.subplot(1,3,2)

sns.countplot(voice_call_oct.Rating)

plt.title('Call Rating in October')



plt.subplot(1,3,3)

sns.countplot(voice_call_nov.Rating)

plt.title('Call Rating in November')



figure.tight_layout(pad = 3.0)
# Lets see what Operators are present in our data for all three months

print(voice_call_sep.Operator.unique())

print(voice_call_oct.Operator.unique())

print(voice_call_nov.Operator.unique())
# Operator count across months



figure = plt.figure(figsize=(20,5))

plt.subplot(1,3,1)

sns.countplot(voice_call_sep.Operator)

plt.title('No of subscribers of respective Operator in September')



plt.subplot(1,3,2)

sns.countplot(voice_call_oct.Operator)

plt.title('No of subscribers of respective Operator in October')



plt.subplot(1,3,3)

sns.countplot(voice_call_nov.Operator)

plt.title('No of subscribers of respective Operator in November')



figure.tight_layout(pad = 3.0)
# Call Drop Category count across months



figure = plt.figure(figsize = (20,4))



plt.subplot(1,3,1)

sns.countplot(voice_call_sep['Call Drop Category'])

plt.title('Call Drop Category in September')



plt.subplot(1,3,2)

sns.countplot(voice_call_oct['Call Drop Category'])

plt.title('Call Drop Category in October')



plt.subplot(1,3,3)

sns.countplot(voice_call_nov['Call Drop Category'])

plt.title('Call Drop Category in Novemeber')



figure.tight_layout(pad=3.0)
# Network Type count across months



figure = plt.figure(figsize = (20,4))



plt.subplot(1,3,1)

sns.countplot(voice_call_sep['Network Type'])

plt.title('No of subscribers of respective Network Type in September')



plt.subplot(1,3,2)

sns.countplot(voice_call_oct['Network Type'])

plt.title('No of subscribers of respective Network Type in October')



plt.subplot(1,3,3)

sns.countplot(voice_call_nov['Network Type'])

plt.title('No of subscribers of respective Network Type in Novemeber')



figure.tight_layout(pad=3.0)
# Lets concatenate the three month data to obr=tain a single dataset. But just in case we later need month related info, we will create a new column 'Month' 

# in all three datasets. We will drop these later if of no help.



voice_call_sep['Month'] = 'September'

voice_call_oct['Month'] = 'October'

voice_call_nov['Month'] = 'November'



frames = [voice_call_sep, voice_call_oct, voice_call_nov]

voice_call = pd.concat(frames)
voice_call.shape
voice_call.info()
# Lets now replace the missing values in the `State Name` column

voice_call['State Name'].fillna('Unknown', inplace = True)



voice_call['State Name'].value_counts()
voice_call = voice_call[~voice_call['State Name'].isin(['California', 'Gangwon-do', 'Lower Saxony', 'Dhaka'])]
plt.figure(figsize = (8,8))

sns.countplot(y = 'State Name', data = voice_call, saturation=1)

plt.title('Number of subscribers from respective States')
plt.figure(figsize = (10,4))



sns.countplot(voice_call.Operator)

plt.title('No of subscribers of respective Operator')



print(voice_call.Operator.value_counts())
voice_call['Operator'] = voice_call['Operator'].replace({'MTNL': 'Others', 'Other': 'Others', 'Telenor': 'Others', 'Tata': 'Others'})

voice_call.Operator.unique()
voice_call.groupby(['Operator'])['Rating'].count()
tempdf=voice_call.groupby(['Operator','Rating'])['Rating'].count().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index(name='percentage')



plt.figure(figsize = (14,4))

ax = sns.barplot(x = tempdf.Operator, y = tempdf.percentage, hue = tempdf.Rating)

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title = 'Rating')
top_state_count_wise = list(voice_call.groupby(['State Name'])['Rating'].count().reset_index().sort_values(by = ['Rating'], ascending = False)['State Name'].head(10))
tempdf = voice_call.groupby(['State Name','Rating'])['Rating'].count().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index(name='percentage')

tempdf = tempdf[tempdf['State Name'].isin(top_state_count_wise)]

plt.figure(figsize = (12, 8))

ax = sns.barplot(y = tempdf['State Name'], x = tempdf.percentage, hue = tempdf.Rating)

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title = 'Rating')

tempdf=voice_call.groupby(['Operator','Call Drop Category'])['Call Drop Category'].count().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index(name='percentage')



plt.figure(figsize = (14,4))

ax = sns.barplot(x = tempdf.Operator, y = tempdf.percentage, hue = tempdf['Call Drop Category'])

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title = 'Call Drop Category')
tempdf=voice_call.groupby(['In Out Travelling','Call Drop Category'])['Call Drop Category'].count().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index(name='percentage')



plt.figure(figsize = (14,4))

ax = sns.barplot(x = tempdf['In Out Travelling'], y = tempdf.percentage, hue = tempdf['Call Drop Category'])

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title = 'Call Drop Category')
tempdf=voice_call.groupby(['Operator','Network Type'])['Network Type'].count().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index(name='percentage')



plt.figure(figsize = (14,4))

ax = sns.barplot(x = tempdf['Operator'], y = tempdf.percentage, hue = tempdf['Network Type'])

ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1, title = 'Network Type')
# Lets just see what the mean value of Latitude and Longitude is.



print(voice_call[voice_call.Latitude != -1.000000].Latitude.mean())

print(voice_call[voice_call.Longitude != -1.000000].Longitude.mean())

# print(np.mode(voice_call[voice_call.Latitude != -1.000000].Latitude))

# print(np.mode(voice_call[voice_call.Longitude != -1.000000].Longitude))
# We shall plot the distribution of Operators across India using Basemap



voice_call_Airtel = voice_call[voice_call.Operator == 'Airtel']

voice_call_BSNL = voice_call[voice_call.Operator == 'BSNL']

voice_call_Idea = voice_call[voice_call.Operator == 'Idea']

voice_call_RJio = voice_call[voice_call.Operator == 'RJio']

voice_call_Vodafone = voice_call[voice_call.Operator == 'Vodafone']

voice_call_Others = voice_call[voice_call.Operator == 'Others']





# Lets also store the lat/ Long coordinate corresponding to Operators.



lon_airtel, lat_airtel = list(voice_call_Airtel[voice_call_Airtel.Longitude != -1.000000].Longitude), list(voice_call_Airtel[voice_call_Airtel.Latitude != -1.000000].Latitude)

lon_bsnl, lat_bsnl = list(voice_call_BSNL[voice_call_BSNL.Longitude != -1.000000].Longitude), list(voice_call_BSNL[voice_call_BSNL.Latitude != -1.000000].Latitude)

lon_idea, lat_idea = list(voice_call_Idea[voice_call_Idea.Longitude != -1.000000].Longitude), list(voice_call_Idea[voice_call_Idea.Latitude != -1.000000].Latitude)

lon_rjio, lat_rjio = list(voice_call_RJio[voice_call_RJio.Longitude != -1.000000].Longitude), list(voice_call_RJio[voice_call_RJio.Latitude != -1.000000].Latitude)

lon_voda, lat_voda = list(voice_call_Vodafone[voice_call_Vodafone.Longitude != -1.000000].Longitude), list(voice_call_Vodafone[voice_call_Vodafone.Latitude != -1.000000].Latitude)

lon_others, lat_others = list(voice_call_Others[voice_call_Others.Longitude != -1.000000].Longitude), list(voice_call_Others[voice_call_Others.Latitude != -1.000000].Latitude)
plt.figure(figsize=(20,12))

# fig.set_size_inches(8, 6.5)



map1 = Basemap(projection='merc',

            llcrnrlat=0, urcrnrlat=38,

            llcrnrlon=60, urcrnrlon=100,

            lat_ts=0,

            resolution='c')



map1.bluemarble(scale=0.2)   # full scale will be overkill

map1.drawcoastlines(color='white', linewidth=0.2)  # add coastlines



x_Airtel, y_Airtel = map1(lon_airtel, lat_airtel)

plt.scatter(x_Airtel, y_Airtel, 10, marker='o', color='#1f77b4', label = 'Airtel')



x_BSNL, y_BSNL = map1(lon_bsnl, lat_bsnl)

plt.scatter(x_BSNL, y_BSNL, 10, marker='o', color='#ff7f0e', label = 'BSNL')



x_Idea, y_Idea = map1(lon_idea, lat_idea)

plt.scatter(x_Idea, y_Idea, 10, marker='o', color='#2ca02c', label = 'Idea')



x_RJio, y_RJio = map1(lon_rjio, lat_rjio) 

plt.scatter(x_RJio, y_RJio, 10, marker='o', color='#d62728', label = 'RJio')



x_Vodafone, y_Vodafone = map1(lon_voda, lat_voda)

plt.scatter(x_Vodafone, y_Vodafone, 10, marker='o', color='#9467bd', label = 'Vodafone')



x_Others, y_Others = map1(lon_others, lat_others)

plt.scatter(x_Others, y_Others, 10, marker='o', color='#8c564b', label = 'Others')        



plt.legend()

plt.show()
# We shall plot the distribution of Ratings across India using Basemap



voice_call_Rating1 = voice_call[voice_call.Rating == 1]

voice_call_Rating2 = voice_call[voice_call.Rating == 2]

voice_call_Rating3 = voice_call[voice_call.Rating == 3]

voice_call_Rating4 = voice_call[voice_call.Rating == 4]

voice_call_Rating5 = voice_call[voice_call.Rating == 5]





# Lets also store the lat/ Long coordinate corresponding to ratings.



lon_rat1, lat_rat1 = list(voice_call_Rating1[voice_call_Rating1.Longitude != -1.000000].Longitude), list(voice_call_Rating1[voice_call_Rating1.Latitude != -1.000000].Latitude)

lon_rat2, lat_rat2 = list(voice_call_Rating2[voice_call_Rating2.Longitude != -1.000000].Longitude), list(voice_call_Rating2[voice_call_Rating2.Latitude != -1.000000].Latitude)

lon_rat3, lat_rat3 = list(voice_call_Rating3[voice_call_Rating3.Longitude != -1.000000].Longitude), list(voice_call_Rating3[voice_call_Rating3.Latitude != -1.000000].Latitude)

lon_rat4, lat_rat4 = list(voice_call_Rating4[voice_call_Rating4.Longitude != -1.000000].Longitude), list(voice_call_Rating4[voice_call_Rating4.Latitude != -1.000000].Latitude)

lon_rat5, lat_rat5 = list(voice_call_Rating5[voice_call_Rating5.Longitude != -1.000000].Longitude), list(voice_call_Rating5[voice_call_Rating5.Latitude != -1.000000].Latitude)
plt.figure(figsize=(20,12))



map2 = Basemap(projection='merc',

            llcrnrlat=0, urcrnrlat=38,

            llcrnrlon=60, urcrnrlon=100,

            lat_ts=0,

            resolution='c')



map2.bluemarble(scale=0.2)   # full scale will be overkill

map2.drawcoastlines(color='white', linewidth=0.2)  # add coastlines



x_rat1, y_rat1 = map2(lon_rat1, lat_rat1)

plt.scatter(x_rat1, y_rat1, 10, marker='o', color='#1f77b4', label = 'Rating 1')



x_rat2, y_rat2 = map2(lon_rat2, lat_rat2)

plt.scatter(x_rat2, y_rat2, 10, marker='o', color='#ff7f0e', label = 'Rating 2')



x_rat3, y_rat3 = map2(lon_rat3, lat_rat3)

plt.scatter(x_rat3, y_rat3, 10, marker='o', color='#2ca02c', label = 'Rating 3')



x_rat4, y_rat4 = map2(lon_rat4, lat_rat4) 

plt.scatter(x_rat4, y_rat4, 10, marker='o', color='#d62728', label = 'Rating 4')



x_rat5, y_rat5 = map2(lon_rat5, lat_rat5)

plt.scatter(x_rat5, y_rat5, 10, marker='o', color='#9467bd', label = 'Rating 5')       



plt.legend()

plt.show()