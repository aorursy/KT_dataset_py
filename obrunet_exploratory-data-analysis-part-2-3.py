import numpy as np

import pandas as pd

from scipy import stats

import seaborn as sns

import matplotlib.pyplot as plt



pd.options.display.max_columns = 300



import warnings

warnings.filterwarnings("ignore")



df_solar_co = pd.read_csv("../input/30-years-of-european-solar-generation/EMHIRESPV_TSh_CF_Country_19862015.csv")

df_solar_co.head(2)
df_solar_co = df_solar_co[['NO', 'AT', 'FR', 'FI', 'RO', 'ES']]

df_solar_co.tail(2)
print("Number of negative values :")

(df_solar_co[['NO', 'AT', 'FR', 'FI', 'RO', 'ES']] < 0).sum()
print("Number of values greater than 1 :")

(df_solar_co[['NO', 'AT', 'FR', 'FI', 'RO', 'ES']] > 1).sum()
def add_time(_df):

    "Returns a DF with two new cols : the time and hour of the day"

    t = pd.date_range(start='1/1/1986', periods=df_solar_co.shape[0], freq = 'H')

    t = pd.DataFrame(t)

    _df = pd.concat([_df, t], axis=1)

    _df.rename(columns={ _df.columns[-1]: "time" }, inplace = True)

    _df['hour'] = _df['time'].dt.hour

    _df['month'] = _df['time'].dt.month

    _df['week'] = _df['time'].dt.week

    return _df



df_solar_co = add_time(df_solar_co)

df_solar_co.tail(2)
def plot_hourly(df, title):

    plt.figure(figsize=(12, 6))

    for c in df.columns:

        if c != 'hour':

            sns.lineplot(x="hour", y=c, data=df, label=c)

            #plt.legend(c)

    plt.title(title)

    plt.show()

    

plot_hourly(df_solar_co[df_solar_co.columns.difference(['time', 'month', 'week'])][-24:], "Efficiency of solar stations per country during the last 24 hours")
plot_hourly(df_solar_co[df_solar_co.columns.difference(['time', 'month', 'week'])], "Mean solar efficiency per country during the day")
temp_df = df_solar_co[df_solar_co.columns.difference(['time', 'hour', 'month', 'week'])]

plt.figure(figsize=(12, 6))

for col in temp_df.columns:

    sns.distplot(temp_df[temp_df[col] != 0][col], label=col, hist=False)

plt.title("Distribution of the station's efficiency for non null values (ie during the day)")
plt.figure(figsize=(12, 6))

sns.lineplot(x = df_solar_co.time, y = df_solar_co['FR'])
countries = ['NO', 'AT', 'FR', 'FI', 'RO', 'ES']



plt.figure(figsize=(12, 6))

for c in countries:

    temp_df = df_solar_co[[c, 'month']]

    sns.lineplot(x=temp_df["month"], y=temp_df[c], label=c)

    

plt.xlabel("Month of year")

plt.ylabel("Efficiency") 

plt.title("Efficiency across the months per country")
plt.figure(figsize=(12, 6))

for c in countries:

    temp_df = df_solar_co[[c, 'week']]

    sns.lineplot(x=temp_df["week"], y=temp_df[c], label=c)

    

plt.xlabel("Week of year")

plt.ylabel("Efficiency") 

plt.title("Efficiency across the weeks per country")
temp_df = df_solar_co.copy()

temp_df['year'] = temp_df['time'].dt.year





plt.figure(figsize=(12, 6))

for c in countries:

    temp_df_ = temp_df[[c, 'year']]

    sns.lineplot(x=temp_df_["year"], y=temp_df_[c], label=c)

    

plt.xlabel("Year")

plt.ylabel("Efficiency") 

plt.title("Efficiency across the years per country")
temp_df = df_solar_co[(5 < df_solar_co.hour) & (df_solar_co.hour < 22)]

temp_df = temp_df.drop(columns=['time', 'hour', 'month', 'week'])

temp_df.describe()
def plot_by_country(_df, title, nb_col):

    _df = _df.describe().iloc[nb_col, :]

    plt.figure(figsize=(10, 6))

    sns.barplot(x=_df.index, y=_df.values)

    plt.title(title)



#plot_by_country("Mean efficiency by country", 1)

plot_by_country(temp_df, "75% efficiency by country", 6)
# credits : https://stackoverflow.com/questions/49554139/boxplot-of-multiple-columns-of-a-pandas-dataframe-on-the-same-figure-seaborn

# This works because pd.melt converts a wide-form dataframe

plt.figure(figsize=(10, 6))

sns.violinplot(x="variable", y="value", data=pd.melt(temp_df))
plt.figure(figsize=(10, 6))

sns.boxplot(x="variable", y="value", data=pd.melt(temp_df))
plt.figure(figsize=(10, 6))

for col in temp_df.columns:

    sns.distplot(temp_df[temp_df[col] != 0][col], label=col, hist=False)

plt.title("Distribution of the station's efficiency")
def plot_corr(df_):

    corr = df_.corr()

    corr



    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Set up the matplotlib figure

    f, ax = plt.subplots(figsize=(9, 7))



    # Generate a custom diverging colormap

    #cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    sns.heatmap(corr, mask=mask, center=0, square=True, cmap='Spectral', linewidths=.5, cbar_kws={"shrink": .5}) #annot=True

    

plot_corr(temp_df)
temp_df.corr()
# credits S Godinho @ https://www.kaggle.com/sgodinho/wind-energy-potential-prediction



df_solar_co['year'] = df_solar_co['time'].dt.year

plt.figure(figsize=(8, 6))

temp_df = df_solar_co[['FR', 'month', 'hour']]

temp_df = temp_df.groupby(['hour', 'month']).mean()

temp_df = temp_df.unstack('month').sort_index(ascending=False)

sns.heatmap(temp_df, vmin = 0.09, vmax = 0.29, cmap = 'plasma')