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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
ne_gp = df[['neighbourhood_group','name','room_type','price','id']]
ne_gp_hotelcount = ne_gp[['neighbourhood_group','id']]
ne_gp_hotelcount = ne_gp_hotelcount.groupby('neighbourhood_group').count()
ne_gp_hotelcount.reset_index(inplace=True)
ne_gp_hotelcount.rename(columns={'id':'Stays_Count'},inplace=True)
ne_gp_hotelcount['Stays_Count_InPercent'] = ne_gp_hotelcount['Stays_Count'] / sum(ne_gp_hotelcount['Stays_Count']) * 100
ne_gp_hotelcount
plt.figure(figsize=(15,10))
plt.pie('Stays_Count_InPercent', labels = 'neighbourhood_group',autopct = '%1.2f%%',data = ne_gp_hotelcount)
plt.title('No.of Stays in Percent by Neighbourhood group')
plt.show()
ne_gp_roomtype = ne_gp[['neighbourhood_group','room_type','id']]
ne_gp_roomtype = ne_gp_roomtype.groupby(['neighbourhood_group','room_type']).count()
ne_gp_roomtype.reset_index(inplace=True)
ne_gp_roomtype.rename(columns = {'id': 'RoomType_Count'}, inplace=True)
ne_gp_roomtype1 = ne_gp_roomtype[['neighbourhood_group','RoomType_Count']]
ne_gp_roomtype1 = ne_gp_roomtype1.groupby('neighbourhood_group').sum()
ne_gp_roomtype1.reset_index(inplace=True)
#ne_gp_roomtype['RoomType_Total'] = ne_gp_roomtype1['RoomType_Count']
ne_gp_roomtype2 = pd.merge(ne_gp_roomtype,ne_gp_roomtype1,on='neighbourhood_group')
ne_gp_roomtype2['RoomType_InPercent'] = ne_gp_roomtype2['RoomType_Count_x'] / ne_gp_roomtype2['RoomType_Count_y'] * 100
ne_gp_roomtype2
plt.figure(figsize=(10,8))
sns.barplot('neighbourhood_group', 'RoomType_InPercent', hue = 'room_type',data = ne_gp_roomtype2)
plt.show()
ne_gp_price = ne_gp[['neighbourhood_group','price']]
ne_gp_price = ne_gp_price.groupby('neighbourhood_group').mean()
ne_gp_price.reset_index(inplace=True)
ne_gp_price['price'] = round(ne_gp_price['price'],2)
ne_gp_price.sort_values('price',ascending=False,inplace=True)  #Sorting based on Price
ne_gp_price
ne_gp_price_dist = ne_gp[['neighbourhood_group','room_type','price']]

#Replacing Private room to P, Entire home/apt to E and Shared room to S
ne_gp_price_dist['room_type'].replace('Private room','P',inplace=True)
ne_gp_price_dist['room_type'].replace('Entire home/apt','E',inplace=True)
ne_gp_price_dist['room_type'].replace('Shared room','S',inplace=True)

#Filter to extract only Bronx neighbourhood group
ne_gp_price_dist_Bronx_P = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Bronx') & (ne_gp_price_dist['room_type'] == 'P')]
ne_gp_price_dist_Bronx_S = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Bronx') & (ne_gp_price_dist['room_type'] == 'S')]
ne_gp_price_dist_Bronx_E = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Bronx') & (ne_gp_price_dist['room_type'] == 'E')]

#Average price of room type in Bronx
print('Average Price of room type in Bronx:')
print('------------------------------------')
print('Average Price of Private Room in Bronx: {}'.format(round(ne_gp_price_dist_Bronx_P.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Entire home/apt in Bronx: {}'.format(round(ne_gp_price_dist_Bronx_E.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Shared Room in Bronx: {}'.format(round(ne_gp_price_dist_Bronx_S.groupby('neighbourhood_group').mean()['price'][0],2)))

print('')
#Median price of room type in Bronx
print('Median Price of room type in Bronx:')
print('------------------------------------')
print('Median Price of Private Room in Bronx: {}'.format(round(ne_gp_price_dist_Bronx_P.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Entire home/apt in Bronx: {}'.format(round(ne_gp_price_dist_Bronx_E.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Shared Room in Bronx: {}'.format(round(ne_gp_price_dist_Bronx_S.groupby('neighbourhood_group').median()['price'][0],2)))


#Filter to extract only Brooklyn neighbourhood group
ne_gp_price_dist_Brooklyn_P = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Brooklyn') & (ne_gp_price_dist['room_type'] == 'P')]
ne_gp_price_dist_Brooklyn_S = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Brooklyn') & (ne_gp_price_dist['room_type'] == 'S')]
ne_gp_price_dist_Brooklyn_E = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Brooklyn') & (ne_gp_price_dist['room_type'] == 'E')]

print('')
#Average price of room type in Brooklyn
print('Average Price of room type in Brooklyn:')
print('----------------------------------------')
print('Average Price of Private Room in Brooklyn: {}'.format(round(ne_gp_price_dist_Brooklyn_P.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Entire home/apt in Brooklyn: {}'.format(round(ne_gp_price_dist_Brooklyn_E.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Shared Room in Brooklyn: {}'.format(round(ne_gp_price_dist_Brooklyn_S.groupby('neighbourhood_group').mean()['price'][0],2)))

print('')
#Median price of room type in Brooklyn
print('Median Price of room type in Brooklyn:')
print('----------------------------------------')
print('Median Price of Private Room in Brooklyn: {}'.format(round(ne_gp_price_dist_Brooklyn_P.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Entire home/apt in Brooklyn: {}'.format(round(ne_gp_price_dist_Brooklyn_E.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Shared Room in Brooklyn: {}'.format(round(ne_gp_price_dist_Brooklyn_S.groupby('neighbourhood_group').median()['price'][0],2)))


#Filter to extract only Manhattan neighbourhood group
ne_gp_price_dist_Manhattan_P = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Manhattan') & (ne_gp_price_dist['room_type'] == 'P')]
ne_gp_price_dist_Manhattan_S = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Manhattan') & (ne_gp_price_dist['room_type'] == 'S')]
ne_gp_price_dist_Manhattan_E = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Manhattan') & (ne_gp_price_dist['room_type'] == 'E')]

print('')
#Average price of room type in Manhattan
print('Average Price of room type in Manhattan:')
print('----------------------------------------')
print('Average Price of Private Room in Manhattan: {}'.format(round(ne_gp_price_dist_Manhattan_P.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Entire home/apt in Manhattan: {}'.format(round(ne_gp_price_dist_Manhattan_E.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Shared Room in Manhattan: {}'.format(round(ne_gp_price_dist_Manhattan_S.groupby('neighbourhood_group').mean()['price'][0],2)))


print('')
#Median price of room type in Manhattan
print('Median Price of room type in Manhattan:')
print('----------------------------------------')
print('Median Price of Private Room in Manhattan: {}'.format(round(ne_gp_price_dist_Manhattan_P.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Entire home/apt in Manhattan: {}'.format(round(ne_gp_price_dist_Manhattan_E.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Shared Room in Manhattan: {}'.format(round(ne_gp_price_dist_Manhattan_S.groupby('neighbourhood_group').median()['price'][0],2)))


#Filter to extract only Queens neighbourhood group
ne_gp_price_dist_Queens_P = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Queens') & (ne_gp_price_dist['room_type'] == 'P')]
ne_gp_price_dist_Queens_S = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Queens') & (ne_gp_price_dist['room_type'] == 'S')]
ne_gp_price_dist_Queens_E = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Queens') & (ne_gp_price_dist['room_type'] == 'E')]

print('')
#Average price of room type in Queens
print('Average Price of room type in Queens:')
print('----------------------------------------')
print('Average Price of Private Room in Queens: {}'.format(round(ne_gp_price_dist_Queens_P.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Entire home/apt in Queens: {}'.format(round(ne_gp_price_dist_Queens_E.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Shared Room in Queens: {}'.format(round(ne_gp_price_dist_Queens_S.groupby('neighbourhood_group').mean()['price'][0],2)))


print('')
#Median price of room type in Queens
print('Median Price of room type in Queens:')
print('----------------------------------------')
print('Median Price of Private Room in Queens: {}'.format(round(ne_gp_price_dist_Queens_P.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Entire home/apt in Queens: {}'.format(round(ne_gp_price_dist_Queens_E.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Shared Room in Queens: {}'.format(round(ne_gp_price_dist_Queens_S.groupby('neighbourhood_group').median()['price'][0],2)))


#Filter to extract only Staten Island neighbourhood group
ne_gp_price_dist_StatenIsland_P = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Staten Island') & (ne_gp_price_dist['room_type'] == 'P')]
ne_gp_price_dist_StatenIsland_S = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Staten Island') & (ne_gp_price_dist['room_type'] == 'S')]
ne_gp_price_dist_StatenIsland_E = ne_gp_price_dist[(ne_gp_price_dist['neighbourhood_group'] == 'Staten Island') & (ne_gp_price_dist['room_type'] == 'E')]

print('')
#Average price of room type in Staten Island
print('Average Price of room type in Staten Island:')
print('----------------------------------------')
print('Average Price of Private Room in Staten Island: {}'.format(round(ne_gp_price_dist_StatenIsland_P.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Entire home/apt in Staten Island: {}'.format(round(ne_gp_price_dist_StatenIsland_E.groupby('neighbourhood_group').mean()['price'][0],2)))
print('Average Price of Shared Room in Staten Island: {}'.format(round(ne_gp_price_dist_StatenIsland_S.groupby('neighbourhood_group').mean()['price'][0],2)))


print('')
#Median price of room type in Staten Island
print('Median Price of room type in Staten Island:')
print('----------------------------------------')
print('Median Price of Private Room in Staten Island: {}'.format(round(ne_gp_price_dist_StatenIsland_P.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Entire home/apt in Staten Island: {}'.format(round(ne_gp_price_dist_StatenIsland_E.groupby('neighbourhood_group').median()['price'][0],2)))
print('Median Price of Shared Room in Staten Island: {}'.format(round(ne_gp_price_dist_StatenIsland_S.groupby('neighbourhood_group').median()['price'][0],2)))


#Quantile price of private room type in Bronx
print('Quantile Price of private room type in Bronx:')
print('------------------------------------')
print('25% of Private room is less than :', np.quantile(ne_gp_price_dist_Bronx_P['price'],0.25))
print('50% of Private room is less than :', np.quantile(ne_gp_price_dist_Bronx_P['price'],0.50))
print('75% of Private room is less than :', np.quantile(ne_gp_price_dist_Bronx_P['price'],0.75))

print('90 Percentile of Private room price is :', np.percentile(ne_gp_price_dist_Bronx_P['price'],90))

print('')

print('Quantile Price of Entire Apt room type in Bronx:')
print('------------------------------------')
print('25% of Entire Apt is less than :', np.quantile(ne_gp_price_dist_Bronx_E['price'],0.25))
print('50% of Entire Apt is less than :', np.quantile(ne_gp_price_dist_Bronx_E['price'],0.50))
print('75% of Entire Apt is less than :', np.quantile(ne_gp_price_dist_Bronx_E['price'],0.75))
print('85 Percentile of Entire Apt price is :', np.percentile(ne_gp_price_dist_Bronx_E['price'],85))

print('')

print('Quantile Price of Shared room type in Bronx:')
print('------------------------------------')
print('25% of Shared room is less than :', np.quantile(ne_gp_price_dist_Bronx_S['price'],0.25))
print('50% of Shared room is less than :', np.quantile(ne_gp_price_dist_Bronx_S['price'],0.50))
print('75% of Shared room is less than :', np.quantile(ne_gp_price_dist_Bronx_S['price'],0.75))
print('95 Percentile of Shared room price is :', np.percentile(ne_gp_price_dist_Bronx_S['price'],95))





