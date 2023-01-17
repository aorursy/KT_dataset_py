import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from matplotlib.dates import DateFormatter

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
test_users = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/test_users.csv')

train_users = pd.read_csv('../input/airbnb-recruiting-new-user-bookings/train_users_2.csv')

train_users.head()
train_users.columns
train_users.shape
test_users.columns
test_users.shape
train_users.dtypes
test_users.dtypes
train_users.isnull().sum() / len(train_users) * 100
test_users.isnull().sum() / len(test_users) * 100
all_users = pd.concat([train_users, test_users])

all_users.shape
all_users.timestamp_first_active = pd.to_datetime(all_users.timestamp_first_active.astype(str), format='%Y%m%d%H%M%S')

all_users.date_account_created = pd.to_datetime(all_users.date_account_created, format='%Y-%m-%d')

all_users.date_first_booking = pd.to_datetime(all_users.date_first_booking, format='%Y-%m-%d')



all_users['year_account_created'] = all_users['date_account_created'].dt.year

all_users['month_account_created'] = all_users['date_account_created'].dt.month

all_users['day_account_created'] = all_users['date_account_created'].dt.day

all_users['weekday_account_created_plot'] = all_users['date_account_created'].dt.day_name()

all_users['weekday_account_created'] = all_users['date_account_created'].dt.dayofweek



all_users['date_first_active'] = all_users['timestamp_first_active'].dt.date

all_users['year_first_active'] = all_users['timestamp_first_active'].dt.year

all_users['month_first_active'] = all_users['timestamp_first_active'].dt.month

all_users['day_first_active'] = all_users['timestamp_first_active'].dt.day



all_users['year_first_booking'] = all_users['date_first_booking'].dt.year

all_users['month_first_booking'] = all_users['date_first_booking'].dt.month

all_users['day_first_booking'] = all_users['date_first_booking'].dt.day

all_users['weekday_first_booking_plot'] = all_users['date_first_booking'].dt.day_name()

all_users['weekday_first_booking'] = all_users['date_first_booking'].dt.dayofweek



#all_users['age'] = all_users['age'].astype('Int64')

all_users.head()
all_users.describe()
from scipy.stats import shapiro

stat, p = shapiro(all_users['age'].dropna())

print('Skewness=%.3f' %all_users['age'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.set(style='whitegrid', rc={'figure.figsize':(8,6), 'axes.labelsize':12})

plt.hist(all_users['age'].dropna(), bins=20);
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

ax1.hist(all_users['age'][all_users['age'] < 10])

ax2.hist(all_users['age'][(all_users['age'] > 100) & (all_users['age'] < 200)])

ax3.hist(all_users['age'][all_users['age'] > 200])

ax4.hist(all_users['age'][(all_users['age'] > 200) & (all_users['age'] < 2000)]);
all_users.age.loc[(all_users['age'] < 10) | (all_users['age'] > 100)] = np.nan

stat, p = shapiro(all_users['age'].dropna())

print('Skewness=%.3f' %all_users['age'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(all_users['age'].dropna());
stat, p = shapiro(train_users['signup_flow'])

print('Skewness=%.3f' %train_users['signup_flow'].skew())

print('Statistics=%.3f, p=%.3f' %(stat, p))



sns.distplot(train_users['signup_flow'], kde=False);
pd.set_option('display.max_rows', 300)

users_cat = all_users.select_dtypes(include=['object']).columns.drop(['id'])



cat_summary = pd.DataFrame()

# loop through categorical variables, and append calculated stats together

for i in range(len(users_cat)):

    c = users_cat[i]

    df = pd.DataFrame({'Variable':[c]*len(all_users[c].unique()),

                       'Level':all_users[c].unique(),

                       'Count':all_users[c].value_counts(dropna=False)})

    df['Percentage'] = 100 * df['Count']  / df['Count'].sum()

    cat_summary = cat_summary.append(df, ignore_index=True)

    

cat_summary
users_by_timeact
users_by_timeact = all_users.groupby(['date_first_active']).size().reset_index(name='n_users')



plt.scatter(users_by_timeact['date_first_active'], users_by_timeact['n_users'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

plt.xticks(rotation=30)

plt.xlabel('Date First Active')

plt.ylabel('Number of Users');
users_by_dateacc = all_users.groupby(['date_account_created']).size().reset_index(name='n_users')



plt.scatter(users_by_dateacc['date_account_created'], users_by_dateacc['n_users'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

plt.xticks(rotation=30)

plt.xlabel('Date Account Created')

plt.ylabel('Number of Users');
users_by_dateacc = all_users.groupby(['date_first_booking']).size().reset_index(name='n_users')



plt.scatter(users_by_dateacc['date_first_booking'], users_by_dateacc['n_users'])

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

plt.xticks(rotation=30)

plt.xlabel('Date First Booking')

plt.ylabel('Number of Users');
all_users['days_btwn_sp_bkng'] = (all_users['date_first_booking'] - all_users['date_account_created']).dt.days

plt.hist((all_users['days_btwn_sp_bkng']).dropna(), bins=20)

plt.xlabel('Days from Signup to First Booking')

plt.ylabel('Number of Users');
all_users['days_btwn_sp_bkng'].describe()
sns.violinplot(x='gender', y='age', palette='Set2', data=all_users);
age_order = all_users.groupby(['country_destination'])['age'].median().reset_index().sort_values('age', ascending=True)



sns.set(style='whitegrid', rc={'figure.figsize':(10,7), 'axes.labelsize':12})

sns.boxplot(x='country_destination', y='age', palette='Set2', data=all_users, order=age_order['country_destination'], linewidth=1.5);
age_order
users_by_ctry_sex = all_users.groupby(['country_destination', 'gender']).size().reset_index(name='n_users')

users_by_ctry_sex_pvt = users_by_ctry_sex.pivot(index='country_destination', columns='gender', values='n_users')

users_by_ctry_sex_pvt_pct = users_by_ctry_sex_pvt.div(users_by_ctry_sex_pvt.sum(1), axis=0)



ax1 = users_by_ctry_sex_pvt.plot(kind='bar')

ax1.set_xlabel('Country Destination')

ax1.set_ylabel('Number of Users')



ax2 = users_by_ctry_sex_pvt_pct.plot(kind='bar')

ax2.set_ylabel('Percentage of Users by Country')

ax2.set_xlabel('Country Destination');
sns.catplot(x="country_destination", y="age", hue="gender", palette='Set2', kind="violin", split=True, 

            data=all_users.loc[all_users['gender'].isin(['FEMALE', 'MALE'])], 

            order=age_order['country_destination'], height=6, aspect=1.5);
users_by_ctry_signup = all_users.groupby(['country_destination', 'signup_method']).size().reset_index(name='n_users')

users_by_ctry_signup_pvt = users_by_ctry_signup.pivot(index='country_destination', columns='signup_method', values='n_users')

users_by_ctry_signup_pvt_pct = users_by_ctry_signup_pvt.div(users_by_ctry_signup_pvt.sum(1), axis=0)



ax1 = users_by_ctry_signup_pvt.plot(kind='bar')

ax1.set_xlabel('Country Destination')

ax1.set_ylabel('Number of Users')



ax2 = users_by_ctry_signup_pvt_pct.plot(kind='bar', stacked=True)

ax2.set_ylabel('Percentage of Users by Country')

ax2.set_xlabel('Country Destination');

ax2.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), title='Signup method');
users_by_ctry_spapp = all_users.groupby(['country_destination', 'signup_app']).size().reset_index(name='n_users')

users_by_ctry_spapp_pvt = users_by_ctry_spapp.pivot(index='country_destination', columns='signup_app', values='n_users')

users_by_ctry_spapp_pvt_pct = users_by_ctry_spapp_pvt.div(users_by_ctry_spapp_pvt.sum(1), axis=0)



ax1 = users_by_ctry_spapp_pvt.plot(kind='bar')

ax1.set_xlabel('Country Destination')

ax1.set_ylabel('Number of Users')



ax2 = users_by_ctry_spapp_pvt_pct.plot(kind='bar', stacked=True)

ax2.set_ylabel('Percentage of Users by Country')

ax2.set_xlabel('Country Destination')

ax2.legend(loc='center right', bbox_to_anchor=(1.2, 0.5), title='Signup app');
signup_flow_order = all_users.groupby(['country_destination'])['signup_flow'].median().reset_index().sort_values('signup_flow', ascending=False)

sns.violinplot(x='country_destination', y='signup_flow', palette='Set2', data=all_users, order=signup_flow_order['country_destination'], linewidth=1.5);
# number of users by country

users_by_ctry = all_users.groupby(['country_destination']).size().reset_index(name='tot_users')

# number of users by country and first device type

users_by_ctry_device = all_users.groupby(['country_destination', 'first_device_type']).size().reset_index(name='n_users')

users_by_ctry_device = users_by_ctry_device.merge(users_by_ctry, left_on='country_destination', right_on='country_destination')

users_by_ctry_device['pct_users'] = users_by_ctry_device['n_users'] / users_by_ctry_device['tot_users']



sns.catplot(x='country_destination', y='pct_users', hue='first_device_type', kind='point', data=users_by_ctry_device, height=6, aspect=1.5);
users_by_ctry_browser = all_users.groupby(['country_destination', 'first_browser']).size().reset_index(name='n_users')

users_by_ctry_browser = users_by_ctry_browser.merge(users_by_ctry, left_on='country_destination', right_on='country_destination')

users_by_ctry_browser['pct_users'] = users_by_ctry_browser['n_users'] / users_by_ctry_browser['tot_users']

users_by_ctry_browser.loc[users_by_ctry_browser.pct_users < 0.01, 'first_browser'] = 'other'

users_by_ctry_browser_agg = users_by_ctry_browser.groupby(['country_destination', 'first_browser'])['pct_users'].sum().reset_index(name='pct_users')

users_by_ctry_browser_pvt = users_by_ctry_browser_agg.pivot(index='country_destination', columns='first_browser', values='pct_users')



ax = users_by_ctry_browser_pvt.plot(kind='bar', stacked=True)

ax.legend(loc='center right', bbox_to_anchor=(1.3, 0.5), title='First browser');
users_by_ctry_affchnl = all_users.groupby(['country_destination', 'affiliate_channel']).size().reset_index(name='n_users')

users_by_ctry_affchnl = users_by_ctry_affchnl.merge(users_by_ctry, left_on='country_destination', right_on='country_destination')

users_by_ctry_affchnl['pct_users'] = users_by_ctry_affchnl['n_users'] / users_by_ctry_affchnl['tot_users']



sns.catplot(x='country_destination', y='pct_users', hue='affiliate_channel', kind='point', data=users_by_ctry_affchnl, height=6, aspect=1.5);
users_by_ctry_affprvdr = all_users.groupby(['country_destination', 'affiliate_provider']).size().reset_index(name='n_users')

users_by_ctry_affprvdr = users_by_ctry_affprvdr.merge(users_by_ctry, left_on='country_destination', right_on='country_destination')

users_by_ctry_affprvdr['pct_users'] = users_by_ctry_affprvdr['n_users'] / users_by_ctry_affprvdr['tot_users']

users_by_ctry_affprvdr.loc[users_by_ctry_affprvdr.pct_users < 0.01, 'affiliate_provider'] = 'other'

users_by_ctry_affprvdr_agg = users_by_ctry_affprvdr.groupby(['country_destination', 'affiliate_provider'])['pct_users'].sum().reset_index(name='pct_users')



sns.catplot(x='country_destination', y='pct_users', hue='affiliate_provider', kind='point', data=users_by_ctry_affprvdr_agg, height=6, aspect=1.5);
users_by_ctry_afftrack = all_users.groupby(['country_destination', 'first_affiliate_tracked']).size().reset_index(name='n_users')

users_by_ctry_afftrack = users_by_ctry_afftrack.merge(users_by_ctry, left_on='country_destination', right_on='country_destination')

users_by_ctry_afftrack['pct_users'] = users_by_ctry_afftrack['n_users'] / users_by_ctry_afftrack['tot_users']

sns.catplot(x='country_destination', y='pct_users', hue='first_affiliate_tracked', kind='point', data=users_by_ctry_afftrack, height=6, aspect=1.5);
users_by_ctry_dateacc = all_users.groupby(['country_destination', 'date_account_created']).size().reset_index(name='n_users')

users_by_ctry_dateacc['date_account_created_plt'] = mdates.date2num(users_by_ctry_dateacc['date_account_created'])



colors = ['windows blue', 'dark salmon', 'amber', 'greyish', 'orchid', 'faded green', 'steel', 'dusty purple',  'olive drab', 'dusty pink', 'sandstone', 'light grey blue']

custompalette = sns.set_palette(sns.xkcd_palette(colors))

sns.scatterplot(x='date_account_created_plt', y='n_users', data=users_by_ctry_dateacc, hue='country_destination', palette=custompalette)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

plt.xticks(rotation=30)

plt.xlabel('Date Account Created')

plt.ylabel('Number of Users');
users_by_ctry_timeact = all_users.groupby(['country_destination', 'date_first_active']).size().reset_index(name='n_users')

users_by_ctry_timeact['date_first_active_plt'] = mdates.date2num(users_by_ctry_timeact['date_first_active'])



sns.scatterplot(x='date_first_active_plt', y='n_users', data=users_by_ctry_timeact, hue='country_destination', palette=custompalette)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.gca().xaxis.set_major_locator(mdates.YearLocator())

plt.gca().xaxis.set_minor_locator(mdates.MonthLocator())

plt.xticks(rotation=30)

plt.xlabel('Date First Active')

plt.ylabel('Number of Users');
all_users.days_btwn_sp_bkng.loc[all_users['days_btwn_sp_bkng'] < 0] = np.nan

sns.violinplot(x='country_destination', y='days_btwn_sp_bkng', data=all_users.loc[all_users['country_destination'] != 'NDF'], palette='deep');
g = sns.FacetGrid(all_users.loc[all_users['country_destination'] != 'NDF'], col='country_destination', col_wrap=4, sharex=False, sharey=False)

g = (g.map(plt.hist, 'month_account_created', color='#4c72b0'));
users_by_ctry_wkdybook = all_users.groupby(['country_destination', 'weekday_account_created_plot']).size().reset_index(name='n_users')



g = sns.FacetGrid(users_by_ctry_wkdybook.loc[users_by_ctry_wkdybook['country_destination'] != 'NDF'], 

                  col='country_destination', col_wrap=4, sharey=False)

g = (g.map(sns.barplot, 'weekday_account_created_plot', 'n_users', 

           order=['Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 

           color='#4c72b0'))

g.set_xticklabels(rotation=30);
g = sns.FacetGrid(all_users.loc[all_users['country_destination'] != 'NDF'], col='country_destination', col_wrap=4, sharex=False, sharey=False)

g = (g.map(plt.hist, 'day_account_created', bins=30, color='#4c72b0'));
g = sns.FacetGrid(all_users.loc[all_users['country_destination'] != 'NDF'], col='country_destination', col_wrap=4, sharex=False, sharey=False)

g = (g.map(plt.hist, 'month_first_booking', color='#4c72b0'));
users_by_ctry_wkdybook = all_users.groupby(['country_destination', 'weekday_first_booking_plot']).size().reset_index(name='n_users')



g = sns.FacetGrid(users_by_ctry_wkdybook.loc[users_by_ctry_wkdybook['country_destination'] != 'NDF'], 

                  col='country_destination', col_wrap=4, sharey=False)

g = (g.map(sns.barplot, 'weekday_first_booking_plot', 'n_users', 

           order=['Monday','Tuesday','Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],

           color='#4c72b0'))

g.set_xticklabels(rotation=30);
g = sns.FacetGrid(all_users.loc[all_users['country_destination'] != 'NDF'], col='country_destination', col_wrap=4, sharex=False, sharey=False)

g = (g.map(plt.hist, 'day_first_booking', bins=30, color='#4c72b0'));