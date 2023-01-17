import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import seaborn as sns

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, LSTM

from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
data = {

    'tra': pd.read_csv('../input/air_visit_data.csv'),

    'as': pd.read_csv('../input/air_store_info.csv'),

    'hs': pd.read_csv('../input/hpg_store_info.csv'),

    'ar': pd.read_csv('../input/air_reserve.csv'),

    'hr': pd.read_csv('../input/hpg_reserve.csv'),

    'id': pd.read_csv('../input/store_id_relation.csv'),

    'tes': pd.read_csv('../input/sample_submission.csv'),

    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})

    }

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

data['hr'].drop('hpg_store_id',  axis=1, inplace=True)

data['ar'] = data['ar'].append(data['hr'])

print ('Data loaded')
air_store_accum = data['tra'].groupby('visit_date', as_index=False).sum().reset_index()



fig, ax = plt.subplots()

fig.set_size_inches(14, 9)

sns.set_style({'axes.linewidth': ".4"})

g = sns.pointplot(data=air_store_accum, x="visit_date", y="visitors", kind='bar', ax=ax,

                  scale=.4)

g.set(xlabel='Date', ylabel='# Visitors')

visit_dates = list(air_store_accum['visit_date'])

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Visitors per date")

plt.show()
data['ar']['visit_date'] = data['ar']['visit_datetime'].str[:10]

ar_reserv_per_rest = data['ar'].groupby(['air_store_id','visit_date'], 

                                        as_index=False).sum().reset_index()

visit_reserv_rel = pd.merge(data['tra'], ar_reserv_per_rest, how='inner',

                                on=['air_store_id', 'visit_date'])

relation_accum = visit_reserv_rel.groupby('visit_date', as_index=False).sum().reset_index()

relation_accum['res_by_vis'] = 100 * relation_accum['reserve_visitors'] / relation_accum['visitors']



fig, ax = plt.subplots()

fig.set_size_inches(14, 9)

sns.set_style({'axes.linewidth': ".4"})

g = sns.pointplot(data=relation_accum, x="visit_date", y="res_by_vis", kind='bar', ax=ax,

                  scale=.4)

g.set(xlabel='Date', ylabel='Reservations by visitors in %')

visit_dates = list(air_store_accum['visit_date'])

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Relationship reservations to visitors")

plt.show()
data['ar']['reserve_date'] = data['ar']['reserve_datetime'].str[:10]

data['ar']['delta_res'] = (pd.to_datetime(data['ar']['visit_date']) - pd.to_datetime(

                              data['ar']['reserve_date'])).dt.days

res_delta_accum = data['ar'].groupby('visit_date', as_index=False).mean().reset_index()



fig, ax = plt.subplots()

fig.set_size_inches(14, 9)

sns.set_style({'axes.linewidth': ".4"})

g = sns.pointplot(data=res_delta_accum , x="visit_date", y="delta_res", kind='bar', ax=ax,

                  scale=.4)

g.set(xlabel='Date', ylabel='Days between reservation and visit')

visit_dates = list(res_delta_accum['visit_date'])

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Days between reservation and visit")

plt.show()
air_area_data = data['tra'].merge(data['as'], on='air_store_id')

air_area_data['air_area_name'] = air_area_data.air_area_name.str[:8]

#differences = [area for area in air_areas if not area in hpg_areas ]

air_area_data = air_area_data.groupby(['air_area_name', 'visit_date'], 

                                      as_index=False).sum().reset_index()

                    

fig, ax = plt.subplots()

fig.set_size_inches(14, 9)

sns.set_style({'axes.linewidth': ".4"})

g = sns.pointplot(data=air_area_data, x="visit_date", y="visitors", kind='bar', ax=ax,

                  hue='air_area_name', scale=.4)

g.set(xlabel='Date', ylabel='# Visitors')

visit_dates = list(air_area_data['visit_date'])

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Visitors per region")

plt.show()
tokio_area_data = data['tra'].merge(data['as'], on='air_store_id')

tokio_area_data['air_area_name'] = air_area_data.air_area_name.str[:8]

tokio_area_data = tokio_area_data[tokio_area_data['air_area_name'] == 'Tōkyō-to']

tokio_area_data = tokio_area_data[pd.to_datetime(tokio_area_data['visit_date']) >= 

                                  pd.to_datetime('2016-02-05')]

tokio_area_data = tokio_area_data[pd.to_datetime(tokio_area_data['visit_date']) <= 

                                  pd.to_datetime('2016-07-15')]



tokio_area_data = tokio_area_data.groupby('visit_date',as_index=False).sum().reset_index()



fig, ax = plt.subplots()

fig.set_size_inches(14, 9)

sns.set_style({'axes.linewidth': ".4"})

g = sns.pointplot(data=tokio_area_data, x="visit_date", y="visitors", kind='bar', ax=ax,

                  scale=.4)

g.set(xlabel='Date', ylabel='# Visitors')

visit_dates = list(tokio_area_data['visit_date'])

for i in range(len(visit_dates)):

    if i % 3 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Visitors in Tokio region around Golden Week 2016")

plt.show()
genres_1 = ('Italian/French', 'Cafe/Sweets',  'Izakaya', 'Dining bar' )

genres_2 = ( 'Bar/Cocktail', 'Other', 'Okonomiyaki/Monja/Teppanyaki', 'Japanese food',

            'Yakiniku/Korean food')

genres_3 = ( 'International cuisine', 'Creative cuisine', 

             'Karaoke/Party', 'Western food', 'Asian')



merged_air_data = data['tra'].merge(data['as'], on='air_store_id')

merged_air_data.drop('latitude',  axis=1, inplace=True)

merged_air_data.drop('longitude',  axis=1, inplace=True)

accum_1 = merged_air_data[merged_air_data['air_genre_name'].isin(genres_1)].groupby(

    ['air_genre_name', 'visit_date'], as_index=False).sum().reset_index()

accum_1.sort_values(by=['visit_date'], inplace=True)

accum_2 = merged_air_data[merged_air_data['air_genre_name'].isin(genres_2)].groupby(

    ['air_genre_name', 'visit_date'], as_index=False).sum().reset_index() 

accum_2.sort_values(by=['visit_date'], inplace=True)

accum_3 = merged_air_data[merged_air_data['air_genre_name'].isin(genres_3)].groupby(

    ['air_genre_name', 'visit_date'], as_index=False).sum().reset_index() 

accum_3.sort_values(by=['visit_date'], inplace=True)
fig, ax = plt.subplots()

fig.set_size_inches(14, 12)

g = sns.pointplot(data=accum_1, x="visit_date", y="visitors", ax=ax, join=True, scale=.4,

                  hue='air_genre_name', palette=sns.color_palette("muted") )

g.set(xlabel='Date', ylabel='# Visitors')   

visit_dates = list(accum_1['visit_date'].drop_duplicates())

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Visitors per restaurant genre (large)")

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(14, 12)

g = sns.pointplot(data=accum_2, x="visit_date", y="visitors", ax=ax, join=True, scale=.4,

                  hue='air_genre_name', palette=sns.color_palette("muted") )

g.set(xlabel='Date', ylabel='# Visitors') 

visit_dates = list(accum_2['visit_date'].drop_duplicates())

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Visitors per restaurant genre (medium)")

plt.show()
fig, ax = plt.subplots()

fig.set_size_inches(14, 12)

g = sns.pointplot(data=accum_3, x="visit_date", y="visitors", ax=ax, join=True, scale=.4,

                  hue='air_genre_name', palette=sns.color_palette("muted") )

g.set(xlabel='Date', ylabel='# Visitors')    

visit_dates = list(accum_3['visit_date'].drop_duplicates())

for i in range(len(visit_dates)):

    if i % 10 != 0: 

        visit_dates[i] = ''  

g.set_xticklabels(visit_dates, rotation=90)

plt.title("Visitors per restaurant genre (small)")

plt.show()