# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("dark")

sns.set(style="ticks", color_codes=True)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dtypes = {"Year": "Int16", "Month": "Int8", "Day": "Int8", "AvgTemperature": "float", "Region" : "category", "Country" : "category", "State" : "category", "City" : "category"}

df = pd.read_csv("/kaggle/input/daily-temperature-of-major-cities/city_temperature.csv", dtype=dtypes)# , nrows=10000



df
print('Check to see that the columns are the right data types. text as category, numbers as floats/ints')

df.info()
df.describe()
regions_covered = ", ".join(df["Region"].unique())

cites_num = len(df["City"].unique())



clean_data = df[(df["AvgTemperature"] > -99)]

first_record =  f'{clean_data["Year"].min()}/{clean_data["Month"].min()}/{clean_data["Day"].min()}'

last_record =  f'{clean_data["Year"].max()}/{clean_data["Month"].max()}/{clean_data["Day"].max()}'



print(f"This data contains a list of daily average temperatures from {cites_num} cities and {len(clean_data['Country'].unique())} countries.")



print(f"The first recorded day is {first_record} and the last {last_record}")

print(f"The data covers the following regions of the world {regions_covered}.")
clean_data = df[(df["AvgTemperature"] > -99)].copy()

# Change Fahrenheit to Celsius

clean_data["AvgTemperature"] = (clean_data["AvgTemperature"] - 32) * (5/9)



clean_data['Date'] = pd.to_datetime(clean_data[['Year', 'Month', 'Day']])

clean_data.index = clean_data["Date"].astype(str) + '/' + clean_data["City"].astype(str)





clean_data = clean_data[clean_data['Date'].dt.year < 2020]

clean_data = clean_data[["Date", "Region", "Country", "State", "City", "AvgTemperature"]]

clean_data.sort_values(by='Date', ascending=True, inplace=True)

clean_data
year_counts = df[df['Year'] > 1000]['Year'].explode().value_counts()



fig, ax = plt.subplots(figsize=(20,2))



ax.barh(year_counts.index, year_counts)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of temperature entries from each year')

ax.set_title('Sample of data point counts by year')



plt.show();
temp_bins = np.histogram(df["AvgTemperature"])



print(f"{temp_bins[0][0]}/{len(df)} entries have temperatues <= {temp_bins[1][0]}")



incomplete_regions = df[df["AvgTemperature"] <= temp_bins[1][0]]["Region"].unique()

print(f'The invalid temperatue entries are from the regions {", ".join(incomplete_regions)}')



incomplete_years = np.array(df[df["AvgTemperature"] <= temp_bins[1][0]]["Year"].unique())

print(f'The invalid temperatue entries are from the years {np.array2string(incomplete_years)}')



plt.figure(figsize=(20,0.5))

plt.title("Distribution of all temperture entries")

ax = df["AvgTemperature"].hist(bins=100)

ax.set_xlabel("Cities temperature °C")

plt.show()
print("""State entries set to null for some cities with only country and no state.

Count of null entries""")

print(df.isnull().sum())
region_counts = clean_data['Region'].explode().value_counts()



plt.rcdefaults()

fig, ax = plt.subplots(figsize=(12,1.5))





ax.barh(region_counts.index, region_counts)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of temperature entries from each region')

ax.set_title('Temperature entries by region')



plt.show();



clean_data["CityRegion"] = clean_data["City"].astype(str) + '/' + clean_data["Region"].astype(str)



city_counts = clean_data['CityRegion'].explode().value_counts()



fig, ax = plt.subplots(figsize=(12,4))



ax.barh(city_counts.index[::15], city_counts[::15])

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Number of temperature entries from each city')

ax.set_title('Temperature entries counts by city')



plt.show();

cites_entries_binned = clean_data['CityRegion'].explode().value_counts().value_counts()

print(f'Median number of temperature entries is {np.median(cites_entries_binned.index)} and spans {round(np.median(cites_entries_binned.index)/365)} years')


plt.figure(figsize=(20,2))

plt.title("All city tempertures binned")

ax = clean_data["AvgTemperature"].hist(bins=1500)

ax.set_xlabel("City daily average °C")

ax.set_ylabel("Count")

plt.show();
clean_data['Days since first measure'] = (clean_data["Date"] - clean_data["Date"].min()).dt.days



piv3 = pd.pivot_table(clean_data, values="AvgTemperature",index=["Days since first measure"], columns=["CityRegion"], fill_value=0)

cmap_cl = sns.diverging_palette(220, 20, s=99, l=50, n=250)



plt.figure(figsize=(19,5))

ax = sns.heatmap(piv3.sort_values(0, axis=1), center=21, cmap=cmap_cl);

for t in ax.texts: t.set_text(t.get_text() + "°C")

city_name = "Seoul" # Washington DC

clean_data['Year'] = clean_data['Date'].dt.year



series_ax = clean_data[clean_data["City"] == city_name].copy()

series_ax["Day of year"] = series_ax["Date"].dt.dayofyear



piv3 = pd.pivot_table(series_ax, values="AvgTemperature",index=["Day of year"], columns=["Year"], fill_value=0)



cmap_cl = sns.diverging_palette(220, 20, as_cmap=True)



plt.figure(figsize=(19,5))

plt.title(f"{city_name} air temperature °C from {piv3.columns[0]} to {piv3.columns[-1]}")

sns.heatmap(piv3, center=21, cmap=cmap_cl, cbar_kws={'label': '°C'})



plt.show()



plt.figure(figsize=(19,5))

plt.title(f"{city_name} change in air temperature °C from {piv3.columns[0]}")

sns.heatmap(piv3.sub(piv3.iloc[:,0], axis=0), center=0, cmap=cmap_cl, cbar_kws={'label': '°C'})

plt.show()



sns.set_style("dark")

plt.figure(figsize=(19,5))

plt.title(f"{city_name} average air temperature °C from {piv3.columns[0]}")

piv3.sub(piv3.iloc[:,0], axis=0).mean().plot(figsize=(19,5), xticks=piv3.columns)



from matplotlib.ticker import FormatStrFormatter

plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f°C'))



plt.show()
