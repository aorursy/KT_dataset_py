# Load useful libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas.core.nanops as nanops # work with NaN in dataframes
import matplotlib.pyplot as plt # plotting
import seaborn as sns; # sophisticated plotting
# Take a peek at the available data
data = pd.read_csv('../input/bucharest-house-prices-september-2020/renting_houses_clean.csv')
data.head(5)
# Get the prices categorised by location and number of rooms, by using the median price for more apartments in same location
area_prices = data.pivot_table(index='location_area', columns='rooms_count', values='price', aggfunc=np.median)

# Create new column '4+' representing mean price of all apartments with 4 or more rooms
area_prices['4+'] = area_prices.loc[:, 4:].mean(axis=1)

# Drop all columns having 4-22 number of rooms as label (those represent sparse data)
area_prices = area_prices.drop(columns=range(4, 23), axis=1, errors='ignore')

# Create new column for mean price of each location, disregarding the number of rooms the apartments own
area_prices['any'] = area_prices.loc[:, [1, 2, 3, '4+']].mean(axis=1)

# Sort by mean price and round prices
area_prices = area_prices.sort_values(by='any', ascending=False).round(2)

# Get relevant information for the most expensive and cheapest areas
expensive_areas = area_prices.head(30)[['any', 1, 2, 3, '4+']]
cheap_areas = area_prices.tail(30).sort_values(by='any')[['any', 1, 2, 3, '4+']]
# Display the final form of the table, computed in the previous block of code
pd.set_option('display.max_rows', None)
area_prices
# Show the most expensive areas in a heatmap
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(expensive_areas, annot=True, linewidths=.5, fmt="g", cmap='Reds', ax=ax)
# Show all (apparently only one) apartments with 1 room in Nordului area
data[(data['location_area'] == 'Nordului') & (data['rooms_count'] == 1)].sort_values(by='price', ascending=False)
# Show the cheapest areas in a heatmap
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(cheap_areas, annot=True, linewidths=.5, fmt="g", cmap='Blues', ax=ax)
# Uverturii has 2 apartments with same prices, but different number of rooms
data[(data['location_area'] == 'Uverturii')]
# Sort areas by median price of single room apartments
single_room_prices = area_prices.sort_values(by=1, ascending=True)

# Drop unnecessary columns, areas with no single room apartments, and keep the first 30
single_room_prices = single_room_prices.drop(columns=[2, 3, '4+', 'any']).dropna()
single_room_prices = single_room_prices.head(30)

# Rename column for convenience
single_room_prices = single_room_prices.rename(columns={1: 'rental_price'})

# Plot the data using a horizontal bar plot
fig, ax = plt.subplots(figsize=(12,12))
ax = sns.barplot(x="rental_price", y=single_room_prices.index, data=single_room_prices, palette="summer")

# Display price annotations for each bar
grouped_values = single_room_prices.reset_index()
for p in ax.patches:
    ax.text(
        x=p.get_x() + p.get_width() + 2,
        y=p.get_y() + p.get_height() * 0.7,
        s=int(p.get_width()),
        ha='left'
    )