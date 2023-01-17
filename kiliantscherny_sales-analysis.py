import pandas as pd

import os

print("Libraries Imported")
df = pd.read_csv("../input/sales-data-for-a-tech-shop/Sales_Data/Sales_April_2019.csv")



# Create an empty pandas dataframe to house all months' data

all_months_data = pd.DataFrame()



# List comprehension to generate a list called 'files'

files = [file for file in os.listdir("../input/sales-data-for-a-tech-shop/Sales_Data")]



# For loop to create a dataframe for each csv file that's in the folder and concatenates it with the all months data 

# dataframe (initially empty) above

for file in files:

    df = pd.read_csv("../input/sales-data-for-a-tech-shop/Sales_Data/"+file)

    all_months_data = pd.concat([all_months_data, df])



# Create a new csv with the data from the dataframe we just created

all_months_data.to_csv("all_data.csv", index=False)
all_data = pd.read_csv("./all_data.csv")

all_data.head()
# Find all instances of NaN in the dataframe

nan_df = all_data[all_data.isna().any(axis=1)]

nan_df.head()
# Drop the NaN rows

all_data = all_data.dropna(how='all')

all_data.head()

# Now our data will be free of these NaN rows
# Here, we're finding all the rows that have the string 'Or' as the first 2 characters in the Order Date column and 

# updating the dataframe to remove them

all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']

all_data.head()



# We can see now that row 1 was removed because it had NaN values
# Make quantity ordered an integer

all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])



# Make price each a float

all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])



all_data.head()
all_data['Month'] = all_data['Order Date'].str[0:2]

all_data['Month'] = all_data['Month'].astype('int32')

all_data.head()
all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

all_data.head()
# Using the .apply() method



# Create a function called 'get_city' that splits the city name

def get_city(address):

    return address.split(',')[1]



# Create a function called 'get_state' that splits the state 2 letter code

def get_state(address):

    return address.split(',')[2].split(' ')[1]



# For each of the cell contents, split by comma and get the city name

all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + ' (' + get_state(x) + ')')



# Alternative way below that uses f strings:

    # all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})")



# deleting the column previously created called 'Column' which was a duplicate of City

# all_data = all_data.drop(columns='Column')



all_data.head()
all_data.groupby('Month').sum()
# Import matplotlib to visualise the data

import matplotlib.pyplot as plt



months = range(1,13)

results = all_data.groupby('Month').sum()



plt.bar(months, results['Sales'])

plt.xticks(months)

plt.ylabel('Sales in USD ($)')

plt.xlabel('Month number')

plt.show()
results = all_data.groupby('City').sum()

results
# List comprehension to ensure cities are in the correct order, to match the above

cities = [city for city, df in all_data.groupby('City')]



plt.bar(cities, results['Sales'])

plt.xticks(cities, rotation='vertical', size=8)

plt.ylabel('Sales in USD ($)')

plt.xlabel('City name')

plt.show()
# Converting the Order Date column to date time format

all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
# Extract hour and minute from Order Date

all_data['Hour'] = all_data['Order Date'].dt.hour

all_data['Minute'] = all_data['Order Date'].dt.minute



all_data.head()
# List comprehension to create a list of the hours, created from the column 'Hour' in the dataframe

hours = [hour for hour, df in all_data.groupby('Hour')]



plt.plot(hours, all_data.groupby(['Hour']).count())

plt.xticks(hours)

plt.xlabel('Hour')

plt.ylabel('Number of Orders')

plt.grid()

plt.show()
# Creating a new dataframe which has all the rows with duplicated Order IDs

df = all_data[all_data['Order ID'].duplicated(keep=False)]



# Creating a new column called Grouped to join up the duplicated products

df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))



# Change the dataframe called df to only have those 2 columns and get rid of duplicate values

df = df[['Order ID', 'Grouped']].drop_duplicates()



df.head()
# Import some new libraries

from itertools import combinations

from collections import Counter
count = Counter()



# Determining the most common combinations of 2 products in an order

for row in df['Grouped']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 2)))



# Get the 10 most common combinations

for key, value in count.most_common(10):

    print(key, value)
all_data.head()
product_group = all_data.groupby('Product')

quantity_ordered = product_group.sum()['Quantity Ordered']



products = [product for product, df in product_group]



plt.bar(products, quantity_ordered)

plt.ylabel('Quantity Ordered')

plt.xlabel('Product')

plt.xticks(products, rotation='vertical', size=8)

plt.show()
prices = all_data.groupby('Product').mean()['Price Each']



fig, ax1 = plt.subplots()



ax2 = ax1.twinx()

ax1.bar(products, quantity_ordered, color='g')

ax2.plot(products, prices, 'b-')



ax1.set_xlabel('Product Name')

ax1.set_ylabel('Quantity Ordered', color='g')

ax2.set_ylabel('Price ($)', color='b')

ax1.set_xticklabels(products, rotation='vertical', size=8)



plt.show()