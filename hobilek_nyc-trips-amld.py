import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import seaborn as sns
# increase the font size of the plots
FONT_SIZE = 14
mpl.rcParams['xtick.labelsize'] = FONT_SIZE 
mpl.rcParams['ytick.labelsize'] = FONT_SIZE
mpl.rcParams['legend.fontsize'] = FONT_SIZE
mpl.rcParams['axes.labelsize'] = FONT_SIZE
mpl.rcParams['figure.figsize'] = (10, 10)
def precision(x, precision):
    """Round a number or array to a given precision
    :Example:

    precision(np.array([3.71, 4.59]), 0.02)
    >> array([3.72, 4.6 ])
    
    precision(np.array([154, 396]), 10)
    >> array([150, 400 ])
    """
    return (np.round(x / precision) * precision).astype(type(precision))

def load_dataset(file_path):
    df = pd.read_csv(file_path, index_col='id', parse_dates=['pickup_datetime'])
    df = df.head(100000)
    # convert the `trip_duration` from seconds to minutes for convenience
    df['trip_duration_minutes'] = df['trip_duration'] / 60.0
    df.drop(['dropoff_datetime', 'store_and_fwd_flag', 'trip_duration'], axis=1, inplace=True)
    return df

def plot_boxplot(df, n_rows, n_cols, figsize=(18, 24)):
    numeric_columns = df.select_dtypes(include=np.number).columns
    fig, axn = plt.subplots(n_rows, n_cols, figsize=figsize)
    for col, ax in zip(numeric_columns, axn.flatten()):
        sns.boxplot(df[col], orient='v', ax=ax)
file_path = '../input/train.csv'

df = load_dataset(file_path)

df.head()
df.describe(percentiles=[0.1, 0.25, 0.75, 0.9])
plot_boxplot(df, n_rows=4, n_cols=2)
fig = plt.figure()
ax = fig.gca()
ax.hist(df['trip_duration_minutes'], bins=50)
ax.set_xlabel('trip duration in min')
ax.set_ylabel('number of trips')
ax.grid()
fig = plt.figure()
ax = fig.gca()
ax.hist(df['trip_duration_minutes'], bins=50)
ax.set_xlabel('trip duration in min')
ax.set_ylabel('number of trips')
ax.set_yscale('log')
ax.grid()
df = df[df['trip_duration_minutes'] < 200]
fig = plt.figure(figsize=(16, 12))
ax = fig.gca()
s = ax.scatter(df['pickup_longitude'], df['pickup_latitude'], marker='.', s=1, 
               c=df['trip_duration_minutes'], cmap='RdYlGn_r')
ax.set_title('pickup locations', fontsize=20)
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.xaxis.set_major_locator(MultipleLocator(0.2))
ax.yaxis.set_major_locator(MultipleLocator(0.2))
ax.grid()
fig.colorbar(s);
df = df[df['pickup_latitude'].between(40.55, 40.95) & \
        df['pickup_longitude'].between(-74.1, -73.7)]
fig = plt.figure(figsize=(16, 12))
ax = fig.gca()
df.plot.hexbin(x='pickup_longitude', y='pickup_latitude', 
               C='trip_duration_minutes', gridsize=20, cmap='Blues', ax=ax);
pickup = df.groupby([precision(df['pickup_longitude'], 0.02), 
                     precision(df['pickup_latitude'], 0.02)]
                   )['trip_duration_minutes'].mean().reset_index().round(2)
pick = pickup.pivot('pickup_latitude','pickup_longitude','trip_duration_minutes')
fig = plt.figure(figsize=(16, 12))
ax = fig.gca()
sns.heatmap(pick, cmap='Greens', ax=ax, annot=True, fmt=".0f")
ax.set_title('mean trip duration [minutes]', fontsize=20)
ax.invert_yaxis()
df['weekday'] = df['pickup_datetime'].dt.weekday # 0 is Monday
df['hour'] = df['pickup_datetime'].dt.hour
df['week_hours'] = df['weekday'] * 24 + df['hour']
week_hours = df.groupby('week_hours')['trip_duration_minutes'].mean()
fig = plt.figure(figsize=(20,6))
ax = fig.gca()
ax.plot(week_hours)
ax.set_xlabel('hours since Monday midnight', fontsize=18)
ax.set_ylabel('mean trip duration [minutes]', fontsize=18)
ax.xaxis.set_major_locator(MultipleLocator(24))
ax.xaxis.set_minor_locator(MultipleLocator(6))
ax.grid(which='minor', linestyle=':', linewidth=0.5)
ax.grid(which='major', linestyle='-', linewidth=1)