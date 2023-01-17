# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pprint

pp = pprint.PrettyPrinter(indent=4)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv("/kaggle/input/nyc-public-wifi/NYC_Wi-Fi_Hotspot_Locations.csv")



x_locations = dataset['Latitude']

y_locations = dataset['Longitude']

from matplotlib.pyplot import scatter

import matplotlib.cm as cm







zipcodes = dataset['Postcode'] # 176 unique zipcodes

colors = cm.gnuplot(np.linspace(0, 1, len(zipcodes)))

scatter(x_locations, y_locations, s=7, c=colors)
manhatten_dataset = pd.read_csv("../input/manhattan-housing-prices/DOF__Cooperative_Comparable_Rental_Income___Manhattan__FY_2011_2012.csv")

manhatten_dataset = manhatten_dataset[manhatten_dataset['Postcode'].notna()]

manhatten_dataset = manhatten_dataset[manhatten_dataset['COMPARABLE RENTAL – 2 – Full Market Value'].notna()]

manhatten_dataset = manhatten_dataset[manhatten_dataset['COMPARABLE RENTAL – 2 – Market Value per SqFt'].notna()]



postcodes = manhatten_dataset['Postcode']

postcodes = [int(i) for i in postcodes]



reduced_x = []

reduced_y = []

reduced_postcodes = []

index = 0

for i in zipcodes:

    if i in postcodes:

        reduced_postcodes.append(zipcodes[index])

        reduced_x.append(x_locations[index])

        reduced_y.append(y_locations[index])

    index += 1

colors = cm.gnuplot(np.linspace(0, 1, len(reduced_postcodes)))

scatter(reduced_x, reduced_y, s=8,c=colors)
zip_full_market_hash = {}

zip_sqft_hash = {}



for zipcode in reduced_postcodes:

    zip_full_market_hash[str(zipcode)] = []

    zip_sqft_hash[str(zipcode)] = []



for index, row in manhatten_dataset.iterrows():

    if str(int(row['Postcode'])) in list(zip_full_market_hash.keys()):

        zip_full_market_hash[str(int(row['Postcode']))].append(row['COMPARABLE RENTAL – 2 – Full Market Value'])

        zip_sqft_hash[str(int(row['Postcode']))].append(row['COMPARABLE RENTAL – 2 – Market Value per SqFt'])
zip_avg_full_market_price = {}

zip_avg_sqft_price = {}



for key in zip_full_market_hash:

    zip_avg_full_market_price[key] = np.mean(np.array(zip_full_market_hash[key]))



for key in zip_sqft_hash:

    zip_avg_sqft_price[key] = np.mean(np.array(zip_sqft_hash[key]))



print("Average full market value per zip code")

pp.pprint(zip_avg_full_market_price)

print("\n")

print("Average price per square foot per zip code")

pp.pprint(zip_avg_sqft_price)
market_price = []

sqft_price = []



for zc in reduced_postcodes:

    market_price.append(zip_avg_full_market_price[str(zc)])

    sqft_price.append(zip_avg_sqft_price[str(zc)])

scatter(reduced_x, reduced_y, s=8,c=market_price, cmap="autumn")

scatter(reduced_x, reduced_y, s=8,c=sqft_price, cmap="autumn")
# Create a hashmap of postal codes to number of hotspots

zipcode_n = {}

for zipcode in reduced_postcodes:

    zipcode_n[str(zipcode)] = 0

    

for index, row in dataset.iterrows():

    if str(int(row['Postcode'])) in list(zipcode_n.keys()):

        zipcode_n[str(int(row['Postcode']))] += 1





# Create 2 columns, one for the price of an area and one for the amount of hotspots in that area

price_col = []

num_col = []





for zipcode in reduced_postcodes:

    z = str(zipcode)

    price_col.append(zip_avg_sqft_price[z])

    num_col.append(zipcode_n[z])



from scipy import stats



pearson_coefficient = stats.pearsonr(price_col, num_col)[0]

print(pearson_coefficient)
scatter(num_col, price_col, s=16)