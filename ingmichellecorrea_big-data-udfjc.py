# Draw inline

%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



# Set figure aesthetics

sns.set_style("white", {'ytick.major.size': 10.0})

sns.set_context("poster", font_scale=1.1)
# Cargue los datos en DataFrames 

ruta =  '../input/' 

train_users = pd.read_csv (ruta +  'train_users_2.csv' )

test_users = pd.read_csv (ruta +  'test_users.csv' )

sessions = pd.read_csv (ruta +  'sessions.csv' )

countries = pd.read_csv (ruta +  'countries.csv' )

age_gender = pd.read_csv (ruta +  'age_gender_bkts.csv' )
print("Tenemos", train_users.shape[0], "registros en el set de entrenamiento y", 

      test_users.shape[0], "en el set de pruebas.")

print("En total tenemos", train_users.shape[0] + test_users.shape[0], "usuarios.")

print(sessions.shape[0], "Registros de sesión para" , sessions.user_id.nunique() , "usuarios." )

print((train_users.shape[0] + test_users.shape[0] -sessions.user_id.nunique()) , "Usuarios sin registros de sessión." )

print((countries.shape[0]) , "Registros en el Dataset de Países." )

print((age_gender.shape[0]) , "registros en el Dataset edad/genero." )
# Unimos usuarios de Pruebas y Entrenamiento

users = pd.concat((train_users, test_users), axis=0, ignore_index=True, sort=False)



# Removemos ID's

users.set_index('id',inplace=True)



users.head()
sessions.head()
countries
users.gender.replace('-unknown-', np.nan, inplace=True)

users.first_browser.replace('-unknown-', np.nan, inplace=True)
users_nan = (users.isnull().sum() / users.shape[0]) * 100

users_nan[users_nan > 0].drop('country_destination')
users.age.describe()
print('Usuarios mayores de 85 años: ' + str(sum(users.age > 85)))

print('Uuarios menores de 18 años: ' + str(sum(users.age < 18)))
users[users.age > 85]['age'].describe()
users[users.age < 18]['age'].describe()
users.loc[users.age > 85, 'age'] = np.nan

users.loc[users.age < 18, 'age'] = np.nan
users.age.describe()
categorical_features = [

    'affiliate_channel',

    'affiliate_provider',

    'country_destination',

    'first_affiliate_tracked',

    'first_browser',

    'first_device_type',

    'gender',

    'language',

    'signup_app',

    'signup_method'

]



for categorical_feature in categorical_features:

    users[categorical_feature] = users[categorical_feature].astype('category')
users['date_account_created'] = pd.to_datetime(users['date_account_created'])

users['date_first_booking'] = pd.to_datetime(users['date_first_booking'])

users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')
users.gender.value_counts(dropna=False).plot(kind='bar', color='#FD5C64', rot=0)

plt.xlabel('Genero')

sns.despine()
women = sum(users['gender'] == 'FEMALE')

men = sum(users['gender'] == 'MALE')



female_destinations = users.loc[users['gender'] == 'FEMALE', 'country_destination'].value_counts() / women * 100

male_destinations = users.loc[users['gender'] == 'MALE', 'country_destination'].value_counts() / men * 100



# Bar width

width = 0.4



male_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=0, label='Male', rot=0)

female_destinations.plot(kind='bar', width=width, color='#FFA35D', position=1, label='Female', rot=0)



plt.legend()

plt.xlabel('Pais de Destino')

plt.ylabel('Porcentaje')



sns.despine()

plt.show()
counts =  users.country_destination.value_counts(normalize=True).plot(kind='bar')

plt.xlabel('Pais de Destino')

plt.ylabel('Porcentaje')
sns.distplot(users.age.dropna(), color='#FD5C64')

plt.xlabel('Age')

sns.despine()
age = 50



younger = sum(users.loc[users['age'] < age, 'country_destination'].value_counts())

older = sum(users.loc[users['age'] > age, 'country_destination'].value_counts())



younger_destinations = users.loc[users['age'] < age, 'country_destination'].value_counts() / younger * 100

older_destinations = users.loc[users['age'] > age, 'country_destination'].value_counts() / older * 100



younger_destinations.plot(kind='bar', width=width, color='#63EA55', position=0, label='Jovenes', rot=0)

older_destinations.plot(kind='bar', width=width, color='#4DD3C9', position=1, label='Viejos', rot=0)



plt.legend()

plt.xlabel('Pais de Destino')

plt.ylabel('Porcentaje')



sns.despine()

plt.show()
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0,1,22))

users[~(users['country_destination'].isin(['NDF']))].groupby(['country_destination' , 'language']).size().unstack().plot(kind='bar', figsize=(20,10),stacked=False,color=colors)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),

          ncol=1, fancybox=True, shadow=True)

plt.yscale('log')

plt.xlabel('Pais de Destino')

plt.ylabel('Log(Conteo)')
sns.set_style("whitegrid", {'axes.edgecolor': '0'})

sns.set_context("poster", font_scale=1.1)

users.date_account_created.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
date_first_active = users.date_first_active.apply(lambda x: datetime.datetime(x.year, x.month, x.day))

date_first_active.value_counts().plot(kind='line', linewidth=1.2, color='#FD5C64')
users['date_account_created'] = pd.to_datetime(users['date_account_created'], errors='ignore')

users['date_first_active'] = pd.to_datetime(users['timestamp_first_active'], format='%Y%m%d%H%M%S')

users['date_first_booking'] = pd.to_datetime(users['date_first_booking'], errors='ignore')
df = users[~users['country_destination'].isnull()]

df.groupby([df["date_account_created"].dt.year, df["date_account_created"].dt.month])['country_destination'].count().plot(kind="bar",figsize=(20,10))
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0,1,12))

df[df["date_first_booking"].dt.year == 2013].groupby(['country_destination' , df["date_first_booking"].dt.month]).size().unstack().plot(kind='bar', stacked=False,color=colors)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),

          ncol=1, fancybox=True, shadow=True)

plt.yscale('log')

plt.xlabel('Destination Country by Month 2013')

plt.ylabel('Log(Count)')
colors = cm.rainbow(np.linspace(0,1,users['affiliate_channel'].nunique()))

users.groupby(['country_destination','affiliate_channel']).size().unstack().plot(kind='bar', stacked=False,color=colors)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),

          ncol=1, fancybox=True, shadow=True)

plt.yscale('log')

plt.xlabel('Destination Country by affiliate channel')

plt.ylabel('Log(Count)')
colors = cm.rainbow(np.linspace(0,1,users['affiliate_provider'].nunique()))

users.groupby(['country_destination','affiliate_provider']).size().unstack().plot(kind='bar', stacked=False,color=colors)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),

          ncol=1, fancybox=True, shadow=True)

plt.yscale('log')

plt.xlabel('Destination Country by affiliate provider')

plt.ylabel('Log(Count)')
colors = cm.rainbow(np.linspace(0,1,users['first_affiliate_tracked'].nunique()))

users.groupby(['country_destination','first_affiliate_tracked']).size().unstack().plot(kind='bar', stacked=False,color=colors)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),

          ncol=1, fancybox=True, shadow=True)

plt.yscale('log')

plt.xlabel('Destination Country by first affiliate tracked')

plt.ylabel('Log(Count)')
import numpy as np

import pandas as pd

users.loc[users.age > 85, 'age'] = np.nan

users.loc[users.age < 18, 'age'] = np.nan

users['age'].fillna(-1,inplace=True)

bins = [-1, 0, 4, 9, 14, 19, 24, 29, 34,39,44,49,54,59,64,69,74,79,84,89]

users['age_group'] = np.digitize(users['age'], bins, right=True)
%matplotlib inline

users.age_group.value_counts().plot(kind='bar')

plt.yscale('log')

plt.xlabel('Age Group')

plt.ylabel('Log(Count)')
df = users[users['country_destination'].isnull()]
date_account_created = pd.DatetimeIndex(users['date_account_created'])

date_first_active = pd.DatetimeIndex(users['date_first_active'])

date_first_booking = pd.DatetimeIndex(users['date_first_booking'])
users['time_lag_create'] = (date_first_booking - date_account_created).days

users['time_lag_active'] = (date_first_booking - date_first_active).days

users['time_lag_create'].fillna(365,inplace=True)

users['time_lag_active'].fillna(365,inplace=True)
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn.apionly as sns

import importlib

importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

ax = sns.boxplot(x="country_destination", y="time_lag_create", showfliers=False,data=users[~(users['country_destination'].isnull())])

#users[~(users['country_destination'].isnull())][['time_lag_create','country_destination']].boxplot(by='country_destination')

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn.apionly as sns

import importlib

importlib.reload(mpl); importlib.reload(plt); importlib.reload(sns)

ax = sns.boxplot(x="country_destination", y="time_lag_active", showfliers=False,data=users[~(users['country_destination'].isnull())])

#users[~(users['country_destination'].isnull())][['time_lag_create','country_destination']].boxplot(by='country_destination')
users[['time_lag_create','time_lag_active']].describe()
users.loc[users.time_lag_create > 365, 'time_lag_create'] = 365

users.loc[users.time_lag_active > 365, 'time_lag_create'] = 365
drop_list = [

    'date_account_created', 'date_first_active', 'date_first_booking', 'timestamp_first_active', 'age'

]



users.drop(drop_list, axis=1, inplace=True)
sessions.rename(columns = {'user_id': 'id'}, inplace=True)
from sklearn import preprocessing

# Create a minimum and maximum processor object

min_max_scaler = preprocessing.MinMaxScaler()



action_count = sessions.groupby(['id'])['action'].nunique()



#action_count = pd.DataFrame(min_max_scaler.fit_transform(action_count.fillna(0)),columns=action_count.columns)

action_type_count = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(len).unstack()

action_type_count.columns = action_type_count.columns.map(lambda x: str(x) + '_count')

#action_type_count = pd.DataFrame(min_max_scaler.fit_transform(action_type_count.fillna(0)),columns=action_type_count.columns)

action_type_sum = sessions.groupby(['id', 'action_type'])['secs_elapsed'].agg(sum)



action_type_pcts = action_type_sum.groupby(level=0).apply(lambda x:

                                                 100 * x / float(x.sum())).unstack()

action_type_pcts.columns = action_type_pcts.columns.map(lambda x: str(x) + '_pct')

action_type_sum = action_type_sum.unstack()

action_type_sum.columns = action_type_sum.columns.map(lambda x: str(x) + '_sum')

action_detail_count = sessions.groupby(['id'])['action_detail'].nunique()



#action_detail_count = pd.DataFrame(min_max_scaler.fit_transform(action_detail_count.fillna(0)),columns=action_detail_count.columns)



device_type_sum = sessions.groupby(['id'])['device_type'].nunique()



#device_type_sum = pd.DataFrame(min_max_scaler.fit_transform(device_type_sum.fillna(0)),columns=device_type_sum.columns)



sessions_data = pd.concat([action_count, action_type_count, action_type_sum,action_type_pcts,action_detail_count, device_type_sum],axis=1)

action_count = None

action_type_count = None

action_detail_count = None

device_type_sum = None





#users = users.join(sessions_data, on='id')
users= users.reset_index().join(sessions_data, on='id')
from sklearn.preprocessing import LabelEncoder

categorical_features = [

    'gender', 'signup_method', 'signup_flow', 'language',

    'affiliate_channel', 'age_group','weekday_account_created','month_account_created','weekday_first_active','month_first_active','hour_first_active',

    'signup_app','affiliate_provider', 'first_affiliate_tracked','first_device_type', 'first_browser'

]

users_sc = users.copy(deep=True)

encode = LabelEncoder()

for j in categorical_features:

    users_sc[j] = encode.fit_transform(users[j].astype('str'))
colx = users_sc.columns.tolist()

rm_list = ['id','country_destination']

for x in rm_list:

    colx.remove(x)

X = users_sc[~(users_sc['country_destination'].isnull())][colx]

X.fillna(0,inplace=True)

from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(0.8))

sel.fit_transform(X)

idxs = sel.get_support(indices=True)

colo = [X.columns.tolist()[i] for i in idxs]

print ('\n'.join(colo))

for y in rm_list:

    colo.append(y)