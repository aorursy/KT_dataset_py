import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
from tqdm import tqdm_notebook as tqdm
import datetime
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv", low_memory = False)
df.sample(5)
df.info(memory_usage='deep')
df.memory_usage(deep=True)
df[['Month', 'Day']] = df[['Month', 'Day']].astype('int8')
df[['Year']] = df[['Year']].astype('int16')
df['AvgTemperature'] = df['AvgTemperature'].astype('float16')
df.reset_index(drop=True, inplace=True)
df.sample(5)
df = df[df.Year != 200]
df = df[df.Year != 201]
df = df[df.Year != 2020]
df = df.drop('State', axis = 1)
df['Date'] = pd.to_datetime(df.Year.astype(str) + '/' + df.Month.astype(str))
missing = pd.DataFrame(df.loc[df.AvgTemperature == -99, 'Country'].value_counts())
missing['TotalData'] = df.groupby('Country').AvgTemperature.count()
missing['PercentageMissing'] = missing.apply(lambda row: (row.Country/row.TotalData)*100, axis = 1)
missing.sort_values(by=['PercentageMissing'], inplace=True, ascending = False)
missing.head(20)
df.loc[df.AvgTemperature == -99, 'AvgTemperature'] = np.nan
df.AvgTemperature.isna().sum()
df['AvgTemperature'] = df['AvgTemperature'].fillna(df.groupby(['City', 'Date']).AvgTemperature.transform('mean'))
df.AvgTemperature.isna().sum()
df.loc[df.AvgTemperature.isna(), 'City'].value_counts()
df['AvgTemperature'] = df['AvgTemperature'].fillna(df.groupby(['City']).AvgTemperature.transform('mean'))
df.AvgTemperature.isna().sum()
# °F to °C: (°F − 32) × 5/9 = °C
df['AvgTempCelsius'] = (df.AvgTemperature -32)*(5/9)
df  = df.drop(['AvgTemperature'], axis = 1)
df['AvgTempCelsius_rounded'] = df.AvgTempCelsius.apply(lambda x: "{0:0.2f}".format(x))
df['AvgTempCelsius_rounded2'] = df.AvgTempCelsius.apply(lambda x: "{0:0.1f}".format(x))
df['AvgTempCelsius_rounded'] = pd.to_numeric(df['AvgTempCelsius_rounded'])
df['AvgTempCelsius_rounded2'] = pd.to_numeric(df['AvgTempCelsius_rounded2'])
df.sample(5)
plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year', y = 'AvgTempCelsius', data = df , palette='Set2')
plt.title('Average Global Temperatures')
plt.ylabel('Average Temperature (°C)')
plt.xlabel('')
plt.xticks(range(1995,2020))
plt.show();
df_mean_month = df.groupby(['Month', 'Year']).AvgTempCelsius_rounded2.mean()
df_mean_month = df_mean_month.reset_index()
df_mean_month = df_mean_month.sort_values(by = ['Year'])
df_pivoted = pd.pivot_table(data= df_mean_month,
                    index='Month',
                    values='AvgTempCelsius_rounded2',
                    columns='Year')
plt.figure(figsize=(20, 8))
sns.heatmap(data = df_pivoted, cmap='coolwarm', annot = True, fmt=".1f", annot_kws={'size':11})
plt.xlabel('')
plt.ylabel('Month')
plt.title('Average Global Temperatures (°C)')
plt.show();
s = df.groupby(['Region'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius',ascending=False)
s.style.background_gradient(cmap="RdBu_r")
f = plt.figure(figsize=(15,8))
sns.lineplot(x = 'Year', y = 'AvgTempCelsius', hue = 'Region', data = df , palette='hsv')
plt.title('Average Temperature in Different Regions')
plt.ylabel('Average Temperature (°C)')
plt.xlabel('Year')
plt.xticks(range(1995,2020))
plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5),ncol=1)
plt.tight_layout()
plt.show();
region_sorted = df.groupby('Region')['AvgTempCelsius'].median().sort_values().index

with sns.color_palette("muted"):
    f, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(data = df.sort_values("AvgTempCelsius"), x = 'Region', y = 'AvgTempCelsius', order = region_sorted)
    plt.xticks(rotation = 90)
    plt.title('Distribution of Temperatures (1995-2019)')
    plt.xlabel('')
    plt.ylabel('Average Temperature (°C)')
with sns.color_palette("muted"):
    f, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(data = df.sort_values("AvgTempCelsius"), x = 'Region', y = 'AvgTempCelsius_rounded', order = region_sorted)
    plt.xticks(rotation = 90)
    plt.title('Distribution of Average Temperatures (1995-2019)')
    plt.xlabel('')
    plt.ylabel('Average Temperature (°C)')
    plt.show;
regions = df.Region.unique().tolist()
import matplotlib.gridspec as gridspec
number_plot = [0, 0, 0, 1, 1, 1, 2]
position_a = [0, 2, 4, 0, 2, 4, 2]
position_b = [2, 4, 6, 2, 4, 6, 4]

fig = plt.figure(figsize = (25,15))
plt.suptitle('Global Monthly Temperatures (1995-2019)', y = 1.05, fontsize=15)

gs = gridspec.GridSpec(3, 6)

for i in range(7): 
    #ax = plt.subplot(3, 3, i+1)
    ax = plt.subplot(gs[number_plot[i], position_a[i]:position_b[i]])
    sns.barplot(x = 'Month', y = 'AvgTempCelsius_rounded2', data = df[df.Region == regions[i]])
    ax.title.set_text(regions[i])
    ax.set_ylim((0,35))
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.subplots_adjust(wspace = 0.5)

plt.savefig('demographics.png')
plt.tight_layout()
plt.show();
s = df.groupby(['Country'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius',ascending=False)[:10]
s.style.background_gradient(cmap="Reds")
s = df.groupby(['Country'])['AvgTempCelsius'].mean().reset_index().sort_values(by='AvgTempCelsius',ascending=True)[:10]
s.style.background_gradient(cmap="Blues")
df_europe = df[df.Region == 'Europe'].copy()
df_europe.sample(5)
f, ax = plt.subplots(figsize=(10, 5))
sns.distplot(df_europe.AvgTempCelsius_rounded, bins = 20)
plt.title('Distribution of Temperatures in Europe (1995-2019)')
plt.xlabel('Temperature (°C)')
#ax.axes.yaxis.set_visible(False)
ax.axes.yaxis.set_ticklabels(['']);
f, ax = plt.subplots(figsize=(10, 5))
sns.distplot(df_europe[df_europe.Year == 2019].AvgTempCelsius_rounded, bins = 20)
plt.title('Distribution of Temperatures in Europe (1995-2019)')
plt.xlabel('Temperature (°C)')
#ax.axes.yaxis.set_visible(False)
ax.axes.yaxis.set_ticklabels(['']);
countries_sorted = df_europe.groupby('Country')['AvgTempCelsius_rounded2'].median().sort_values().index

with sns.color_palette("muted"):
    f, ax = plt.subplots(figsize=(20, 7))
    sns.boxplot(data = df_europe, x = 'Country', y = 'AvgTempCelsius_rounded', order = countries_sorted)
    plt.xticks(rotation = 90)
    plt.title('Distribution of Temperatures in Europe (1995-2019)')
    plt.ylabel('Temperature (°C)')
    plt.xlabel('');
countries_mean_sorted = df_europe.groupby('Country').AvgTempCelsius_rounded2.mean().sort_values().index

plt.figure(figsize = (15,8))
sns.barplot(x = 'Country', y = 'AvgTempCelsius_rounded2', data = df_europe, 
            order = countries_mean_sorted)
plt.xticks(rotation = 90)
plt.xlabel('')
plt.title('Average Temperatures in Europe (1995-2019)')
plt.ylabel('Average Temperature (°C)');
plt.figure(figsize = (15,8))
sns.barplot(x = 'Year', y = 'AvgTempCelsius_rounded2', data = df_europe)
plt.title('Average Yearly Temperature in Europe')
plt.xlabel('Year')
plt.ylabel('Average Temperature (°C)')
plt.show();
europe_mean_month = df_europe.groupby(['Month', 'Year']).AvgTempCelsius_rounded2.mean()
europe_mean_month = europe_mean_month.reset_index()
europe_mean_month = europe_mean_month.sort_values(by = ['Year'])
europe_pivoted = pd.pivot_table(data= europe_mean_month,
                    index='Month',
                    values='AvgTempCelsius_rounded2',
                    columns='Year')
plt.figure(figsize=(20, 8))
sns.heatmap(data = europe_pivoted, cmap='coolwarm', annot = True, fmt=".1f", annot_kws={'size':11})
plt.ylabel('Month')
plt.xlabel('')
plt.title('Average Temperatures in Europe (°C)')
plt.show();
df_austria = df_europe[df_europe.Country == 'Austria'].copy()
df_austria.head()
f, ax = plt.subplots(figsize=(10, 7))
sns.distplot(df_austria.AvgTempCelsius_rounded, bins = 20);
plt.title('Distribution of Average Temperatures in Austria (1995-2019)')
plt.xlabel('Average Temperature (°C)')
ax.axes.yaxis.set_ticklabels([]);
plt.figure(figsize=(20,8))
sns.lineplot(x = 'Year', y = 'AvgTempCelsius_rounded2', data = df_austria , palette='hsv')
plt.title('Average Temperatures in Austria')
plt.ylabel('Average Temperature (°C)')
plt.xlabel('')
plt.xticks(range(1995,2020))
plt.show()
years = df_austria.Year.unique().tolist()
years = [str(year) for year in years]
plt.figure(figsize=(20,8))
sns.lineplot(x = 'Month', y = 'AvgTempCelsius_rounded2', data = df_austria , palette='hsv')
plt.title('Average Monthly Temperatures in Austria (1995-2019)')
plt.ylabel('Average Temperature (°C)')
plt.xlabel('Month')
plt.xticks(range(1,13))
plt.show();
months = ['January', 'February', 'March', 'April', 'May', 'June', 
          'July', 'August', 'September', 'October', 'November', 'December']
fig = plt.subplots(3,4, figsize = (15,8))
for i in range(1,13): 
    ax = plt.subplot(3, 4, i)
    sns.lineplot(x = 'Day', y = 'AvgTempCelsius_rounded2', data = df_austria[df_austria.Month == i] , palette='hsv')
    ax.title.set_text(months[i-1])
    ax.set_ylim((-5,25))
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.suptitle('Monthly Temperatures in Austria (1995-2019)', y = 1.05)
#plt.ylabel('Average Temperature (°C)')
plt.tight_layout()
plt.show();
austria_pivoted = pd.pivot_table(data= df_austria,
                    index='Month',
                    values='AvgTempCelsius_rounded2',
                    columns='Year')
plt.figure(figsize=(20, 8))
sns.heatmap(data = austria_pivoted, cmap='coolwarm', annot = True, fmt=".1f")
plt.ylabel('Month')
plt.xlabel('')
plt.title('Average Temperatures in Austria (°C)')
plt.show();