import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

plt.style.use('bmh')

# mpl.rcParams['axes.labelcolor'] = 'grey'

# mpl.rcParams['xtick.color'] = 'grey'

mpl.rcParams['xtick.labelsize'] = 'large'

# mpl.rcParams['ytick.color'] = 'grey'

# mpl.rcParams['axes.labelcolor'] = 'grey'

# mpl.rcParams['text.color'] = 'grey'

mpl.rcParams["legend.loc"] = 'best'
buildings = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_train.csv")

weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")
buildings['age'] = 2016 - buildings['year_built']
sns.distplot(buildings.age.dropna());
sns.distplot(buildings.square_feet.dropna());
def show_counts(df, col, n):

    cnt = df[col].value_counts().nlargest(n)[::-1]

    fig, ax = plt.subplots()

    ax.barh(cnt.index.astype(str), cnt.values)

    ax.set_xlabel('count')

    ax.set_title(col, color='white')

    for i, v in enumerate(cnt):

        ax.text(v + 3, i, str(int(v/cnt.sum() * 100)) + "%", color='grey', fontweight='bold')
show_counts(buildings, 'primary_use', 5);
show_counts(buildings, 'floor_count', 7)
def compare_kde(df, col, n):

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(9, 10), sharex=True)

    most_use_case = buildings.primary_use.value_counts().nlargest(n).index

    for i, use_case in enumerate(most_use_case):

        sns.distplot(buildings.loc[buildings.primary_use==use_case, col].dropna(), ax=axes[i]);

        axes[i].set_title(use_case)

    plt.tight_layout()
compare_kde(buildings, 'age', 5)
compare_kde(buildings, 'floor_count', 5)
compare_kde(buildings, 'square_feet', 5)
weather_train.head()
weather_sites = weather_train.site_id.unique()
calculate_nans = ['site_id', 'air_temperature', 'cloud_coverage', 'dew_temperature',

                  'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

                  'wind_speed']

weather_concat = pd.concat([weather_train, weather_test], axis=0)

calculate_nans_df = weather_concat[calculate_nans].isna()

calculate_nans_df['site_id'] = weather_concat.site_id

calculate_nans_df = calculate_nans_df.groupby('site_id').sum() / (24 * 365 * 3) * 100
# add time attribute

# Only Month, HourofDay have something to do with weather

weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])

weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])



weather_train['hour'] = weather_train.timestamp.dt.hour

weather_test['hour'] = weather_test.timestamp.dt.hour



weather_train['month'] = weather_train.timestamp.dt.month

weather_test['month'] = weather_test.timestamp.dt.month



weather_train = weather_train.set_index('timestamp')

weather_test = weather_test.set_index('timestamp')
def plot_weather_attribute(col, rolling_window=60):

    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 12))

    exclude_site = [c for c in weather_train.columns if c != 'site_id']

    rolling_weather_train = weather_train[exclude_site].rolling(rolling_window).mean()

    rolling_weather_train['site_id'] = weather_train.site_id



    rolling_weather_test = weather_test[exclude_site].rolling(rolling_window).mean()

    rolling_weather_test['site_id'] = weather_test.site_id



    for i in range(6):

        for j in range(3):

            site_idx = i * 3 + j

            if (site_idx) > 15:

                break

            rolling_weather_train.loc[rolling_weather_train.site_id==(site_idx), col].plot(ax=axes[i, j])

            rolling_weather_test.loc[rolling_weather_test.site_id==(site_idx), col].plot(ax=axes[i, j], color='green')



            axes[i, j].set_title(f"Site {site_idx} NULL: {int(calculate_nans_df.loc[site_idx, col])}%")

    plt.tight_layout()
plot_weather_attribute('air_temperature', rolling_window=60)
plot_weather_attribute('cloud_coverage', rolling_window=7)
plot_weather_attribute('dew_temperature', rolling_window=60)
plot_weather_attribute('precip_depth_1_hr', rolling_window=7)
plot_weather_attribute('sea_level_pressure', rolling_window=7)
plot_weather_attribute('wind_direction', rolling_window=60)
plot_weather_attribute('wind_speed', rolling_window=60)
train_site_hour_mean = weather_train.groupby(['site_id', 'hour']).mean()

test_site_hour_mean = weather_test.groupby(['site_id', 'hour']).mean()
def plot_weather_by_hour(col):

    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 12))

    for i in range(6):

        for j in range(3):

            site_idx = (i * 3 + j)

            if site_idx > 15:

                break

            train_site_data = train_site_hour_mean.loc[(site_idx, slice(None)), col]

            train_site_data.index = train_site_data.index.droplevel(0)



            test_site_data = test_site_hour_mean.loc[(site_idx, slice(None)), col]

            test_site_data.index = test_site_data.index.droplevel(0)



            train_site_data.plot(ax=axes[i, j])

            test_site_data.plot(ax=axes[i, j], color='green')



            axes[i, j].set_title(f"Site {site_idx} NULL: {int(calculate_nans_df.loc[site_idx, col])}%")

    plt.tight_layout()
weather_train.columns
plot_weather_by_hour('air_temperature')
plot_weather_by_hour('cloud_coverage')
check_cloud_na = weather_train[['site_id', 'hour', 'cloud_coverage']].copy()

check_cloud_na.loc[:, 'cloud_coverage'] = check_cloud_na['cloud_coverage'].isna()

check_cloud_na.groupby(['site_id', 'hour']).sum().unstack('hour')
plot_weather_by_hour('dew_temperature')
plot_weather_by_hour('precip_depth_1_hr')
plot_weather_by_hour('sea_level_pressure')
plot_weather_by_hour('wind_direction')
plot_weather_by_hour('wind_speed')
train_site_month_mean = weather_train.groupby(['site_id', 'month']).mean()

test_site_month_mean = weather_test.groupby(['site_id', 'month']).mean()
def plot_weather_by_month(col):

    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 12))

    for i in range(6):

        for j in range(3):

            site_idx = (i * 3 + j)

            if site_idx > 15:

                break

            train_site_data = train_site_month_mean.loc[(site_idx, slice(None)), col]

            train_site_data.index = train_site_data.index.droplevel(0)



            test_site_data = test_site_month_mean.loc[(site_idx, slice(None)), col]

            test_site_data.index = test_site_data.index.droplevel(0)



            train_site_data.plot(ax=axes[i, j])

            test_site_data.plot(ax=axes[i, j], color='green')



            axes[i, j].set_title(f"Site {site_idx} NULL: {int(calculate_nans_df.loc[site_idx, col])}%")

    plt.tight_layout()
plot_weather_by_month('air_temperature')
plot_weather_by_month('cloud_coverage')
plot_weather_by_month('dew_temperature')
plot_weather_by_month('precip_depth_1_hr')
plot_weather_by_month('sea_level_pressure')
plot_weather_by_month('wind_direction')
plot_weather_by_month('wind_speed')
def plot_weather_correlation(df):

    fig, axes = plt.subplots(nrows=6, ncols=3, sharex=True, sharey=True, figsize=(18, 24))

    for i in range(6):

        for j in range(3):

            site_idx = (i * 3 + j)

            if site_idx > 15:

                for tick in axes[i, j].get_xticklabels():

                    tick.set_rotation(90)

                continue

                

            sns.heatmap(df.loc[df.site_id==site_idx, 

                                          ['air_temperature', 'cloud_coverage', 'dew_temperature',

                                          'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

                                          'wind_speed']

                                         ]\

                                         .corr(), ax=axes[i, j], annot=True, fmt ='.1f')

            axes[i, j].set_title(f"Site {site_idx}")

    plt.tight_layout()
sns.heatmap(weather_train[['air_temperature', 'cloud_coverage', 'dew_temperature',

               'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

               'wind_speed']].corr(), annot=True, fmt ='.2f');
sns.heatmap(weather_test[['air_temperature', 'cloud_coverage', 'dew_temperature',

               'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',

               'wind_speed']].corr(), annot=True, fmt ='.2f');
plot_weather_correlation(weather_train)
# In Test Set by Site
plot_weather_correlation(weather_test)