import numpy as np # NumPy is the fundamental package for scientific computing



import pandas as pd # Pandas is an easy-to-use data structures and data analysis tools

pd.set_option('display.max_columns', None) # To display all columns



import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library

%matplotlib inline 

# A magic command that tells matplotlib to render figures as static images in the Notebook.



import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).

sns.set_style('whitegrid') # One of the five seaborn themes

import warnings

warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg



from scipy import stats, linalg



from matplotlib import rcParams

import scipy.stats as st



import folium # for map visualization

from folium import plugins
train = pd.read_csv("../input/2019-2nd-ml-month-with-kakr/train.csv", parse_dates=['date'])

test = pd.read_csv("../input/2019-2nd-ml-month-with-kakr/test.csv", parse_dates=['date'])
# Why are the following features converted to category type?

train['waterfront'] = train['waterfront'].astype('category',ordered=True)

train['view'] = train['view'].astype('category',ordered=True)

train['condition'] = train['condition'].astype('category',ordered=True)

train['grade'] = train['grade'].astype('category',ordered=False) # Why are these ordered 'False'?

train['zipcode'] = train['zipcode'].astype(str)

train.head(2) # Show the first 2 lines
print(train.shape)

print(train.nunique())
print(train.info())
# Knowing the Price variable

f, ax = plt.subplots(1, 2, figsize = (14, 6))

sns.distplot(train['price'], ax=ax[0])

ax[0].set_title('Price Distribution')

plt.xlim()



sns.scatterplot(range(train.shape[0]), train['price'].sort_values(), ax=ax[1], marker="x")

ax[1].set_title('Price Curve Distribution', fontsize=15)



plt.show()
price_des = train.describe()['price']

price_des.astype('int')
houses_map = folium.Map(location = [train['lat'].mean(), train['long'].mean()], zoom_start = 10)

lat_long_data = train[['lat', 'long']].values.tolist()

h_cluster = folium.plugins.FastMarkerCluster(lat_long_data).add_to(houses_map)



houses_map
houses_heatmap = folium.Map(location = [train['lat'].mean(), train['long'].mean()], zoom_start=9)

houses_heatmap.add_children(plugins.HeatMap([[row['lat'], row['long']] for name, row in train.iterrows()]))

houses_heatmap
zipcode_data = train.groupby('zipcode').aggregate(np.mean)
zipcode_data.reset_index(inplace=True)
train['count'] = 1

count_house_zipcode = train.groupby('zipcode').sum()

count_house_zipcode.reset_index(inplace=True)

count_house_zipcode = count_house_zipcode[['zipcode', 'count']]

train.drop(['count'], axis = 1, inplace=True)
zipcode_data = zipcode_data.join(count_house_zipcode.set_index('zipcode'), on='zipcode')
zipcode_data.head()
def show_zipcode_map(col, bins):

    geo_path = '../input/house-prices-data/zipcode_king_county.geojson'

    zipcode = folium.Map(location=[train['lat'].mean(), train['long'].mean()], zoom_start=9)

    zipcode.choropleth(

        geo_data=geo_path, 

        data=zipcode_data, 

        columns=['zipcode', col],

        key_on='feature.properties.ZCTA5CE10',

        fill_color='OrRd',

        fill_opacity=0.6, 

        line_opacity=0.2,

        bins=bins) # bins로 조절!

#     zipcode.save(col + '.html')

    return zipcode
show_zipcode_map('count', 9)
show_zipcode_map('price', 7)
show_zipcode_map('sqft_lot', 9)
show_zipcode_map('sqft_lot15', 9)
show_zipcode_map('sqft_living', 9)
show_zipcode_map('sqft_living15', 9)
plt.figure(figsize=(12,6))

sns.distplot(np.log(train['sqft_lot']), hist=False, color='r', label='lot')

sns.distplot(np.log(train['sqft_lot15']), hist=False, color='b', label='lot15')

plt.show()
plt.figure(figsize=(12,6))

sns.distplot((train['sqft_living']), hist=False, color='r', label='living')

sns.distplot((train['sqft_living15']), hist=False, color='b', label='living15')

plt.show()
plt.figure(figsize=(12,6))

sns.scatterplot(x=train.index, y='sqft_lot', data=train, color='r', label='lot')

sns.scatterplot(x=train.index, y='sqft_lot15', data=train, color='b', label='lot15')

plt.show()
plt.figure(figsize=(12,6))

sns.scatterplot(x=train.index, y=train['sqft_living'], color='r', label='living')

sns.scatterplot(x=train.index, y=train['sqft_living15'], color='b', label='living15')

plt.show()
round(train.describe()[['sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15']], 0)
round(train.describe().loc[['75%', 'max'], ['sqft_living', 'sqft_living15', 'sqft_lot', 'sqft_lot15']], 0)
plt.figure(figsize=(12,6))

sns.boxplot(data=train[['sqft_living', 'sqft_living15']], orient='h')

plt.show()
plt.figure(figsize=(18,6))

sns.boxplot(data=np.log(train[['sqft_lot', 'sqft_lot15']]), orient='h')

plt.show()