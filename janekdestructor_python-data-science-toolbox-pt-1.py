import numpy as np

from random import randint

from datetime import datetime



# testing performance difference between python lists and numpy array

def get_list_with_x_random_elements(x):

    return [randint(1, 9) for i in range(x)]



list_a = get_list_with_x_random_elements(1000000)

list_b = get_list_with_x_random_elements(1000000)

np_array_a = np.array(list_a)

np_array_b = np.array(list_b)



def calculate_performance_for_python_list():

    dt1 = datetime.now()

    result = [(a / b) ** 2 for a, b in zip(list_a, list_b)]

    dt2 = datetime.now()

    return (dt2 - dt1).microseconds



def calculate_performance_for_numpy_array():

    dt1 = datetime.now()

    result = (np_array_a / np_array_b) ** 2

    dt2 = datetime.now()

    return (dt2 - dt1).microseconds
python_list_result = [calculate_performance_for_python_list() for i in range(20)]

numpy_array_result = [calculate_performance_for_numpy_array() for i in range(20)]
# Calculating mean for both results

print("Python list performance:", np.array(python_list_result).mean())

print("NumPy array performance:", np.array(numpy_array_result).mean())
np.array([1, 2.4, "string", False])
a = np.array([1, 2, 4, 8, 11])

bolean_filter = a > 10

bolean_filter
a[bolean_filter]
# Replacing values based on logical operations

np.where(a > 10, 1, 0)
# Generating random normal distribution 

normal_dist = np.random.normal(100, 5, 10)

# arguments in order: mean, stadard deviation and number of samples

normal_dist
normal_dist = np.round(normal_dist, 2)

normal_dist
print("Mean:", np.mean(normal_dist))

print("Median:", np.median(normal_dist))

print("Standard deviation:", np.std(normal_dist))

print("Variance:", np.var(normal_dist))
# Relation between two sets

normal_dist2 = np.random.normal(200, 10, 10)

print("Correlation coefficients:")

print(np.corrcoef(normal_dist, normal_dist2))
# Methods which are also avialable for standard python lists, but work much faster

print("Max value:", np.max(normal_dist))

print("Min value:", np.min(normal_dist))

print("Index of min value:", np.argmin(normal_dist))

print("Index of max value:", np.argmax(normal_dist))

print("Sum:", np.sum(normal_dist))

print("Product:", np.prod(normal_dist))

print("Sorted:", np.sort(normal_dist))
array_1 = np.array([11,12,13,14,15,16,17,18,19])

array_2 = np.array([10,20,30,40,50,60,70,80,90])
print("Addition:", array_1 + array_2)

print("Subtraction:", array_2 - array_1)

print("Multiplication:", array_1 * array_2)

print("Division:", array_1 / array_2)
array_1 + 100 
# Generating NumPy arrays with zeros or ones

zeros = np.zeros(10)

ones = np.ones(10)



print(zeros)

print(ones)
ones.shape
ones.shape = 10, 1

ones
# Generasting array with constant step

np.linspace(2, 100, 8)

# arguments: start, end, number of elements
# dot product (iloczyn skalarny)

array_1 @ array_2
list_A = [100, 46, 45, 82, 90]

list_B = [404, 24, 87, 99, 12]
import pandas as pd



# Creating Data Frame from python dictionary

dict_data = {

    "country": ["Poland", "Germany", "Spain", "Denmark"],

    "capital": ["Warsaw", "Berlin", "Madrid", "Copenhagen"],

    "area": [312679, 357386, 505990, 42933],

    "population": [38.4, 82.8, 46.7, 5.7]

}



df = pd.DataFrame(dict_data)

df
# Pandas assigns index (0,1,2,3) automatically but we can set index manually

df.set_index('country')
# Importing data from a CSV file

df = pd.read_csv('../input/otomoto.csv')



df = df[df.make == 'Tesla'].head(10).reset_index()

# remove column 'index'

del df['index']

df
# Selecting specific column

price_column = df["price"]

price_column
type(price_column)
# Syntax for selecting specific column but in a format of a DataFrame

price_column_df = df[["price"]]

price_column_df
type(price_column_df)
# Selecting multiple columns

price_year_df = df[['price', 'year']]

price_year_df
# Selecting rows from a data frame where car is less expensive than 200 000 PLN

cheap_tesla = df[df.price < 200000]

cheap_tesla
# Using logical AND when filtering

filtering_mask = np.logical_and(df.price < 200000, df.year == 2015)

df[filtering_mask]
# Using logical OR when filtering

filtering_mask = np.logical_or(df.price < 200000, df.mileage < 10000)

df[filtering_mask]
# Iteratig over data frame with a for loop

for label, row in cheap_tesla.iterrows():

    print("Label:", label)

    print("Row values:")

    print(row)

    

# Remember! This is not very efficient
# Assuming we want to create new column which would be a price in EUR

df["price_EUR"] = np.round(df["price"] / 3.9, 2)
# We can apply any user defined function for each element with 'apply'

def my_func(x):

    return np.round(x / 3.9, 2)



df["price_EUR"] = df["price"].apply(my_func)

df
# Collecting basic info about a DataFrame

df = pd.read_csv('../input/otomoto.csv')

pd.options.display.float_format = '{:.2f}'.format
df.shape
df.columns
# Basic info about each column

df.info()
# Basic statistical information about numerical columns

df.describe()
# See top rows of a DataFrame

df.head()
# See bottom rows of a DataFrame

df.tail()
# Sort DataFrame

df = df.sort_values("price")

df.head()
# Getting rid of currency column - converting all rows to PLN currency

eur_pln_ratio = 4.31

df['price'] = df.apply(lambda x: int(x['price'] * eur_pln_ratio) if x['currency'] == 'EUR' else x['price'], axis=1)

df.loc[70123]

# remove currency column

del df['currency']
# Explore prices distribution

df.price.plot('hist')

from pylab import rcParams

rcParams['figure.figsize'] = 7,7
df.sort_values("price", ascending=False).head(20)
# Reading standard deviation

np.std(df.price)
# delete all rows where price is bigger than 200 000 PLN

df = df[df.price < 200000]
# Plotting the histogram once again

df.price.plot('hist')
# delete all rows where price is lower than 2000 PLN

df = df[df.price > 2000]
# Plotting the histogram once again

df.price.plot('hist')
# Exploring makes

len(df.make.value_counts())
pd.set_option('display.max_rows', 100)
df.make.value_counts()
df[df.make == "Samochód"]
# delete all rows where make is equal to 'Samochód'

df = df[df.make != 'Samochód']
# delete all rows with make that has less than 500 occurances

v = df[['make']]

df = df[v.replace(v.apply(pd.Series.value_counts)).gt(500).all(1)]

len(df.make.value_counts())
# Exploring fuel column

df.fuel.value_counts()
df = df[np.logical_and(df.fuel != "Wodór", df.fuel != 'Etanol')]

df.fuel.value_counts()
import matplotlib.pyplot as plt

# correlation between year and price

_ = plt.scatter(df['price'], df['year'])

plt.show()
# remove all rows where year is below 1970

df = df[df.year > 1970]
_ = plt.scatter(df['price'], df['year'])

plt.show()
# correlation between price and mileage

_ = plt.scatter(df['price'], df['mileage'])

plt.show()
# remove all rows where mileage is above 700 000

df = df[df.mileage < 700000]
# correlation between price and mileage

_ = plt.scatter(df['price'], df['mileage'])

plt.show()
# correlation between year and mileage

_ = plt.scatter(df['year'], df['mileage'])

plt.show()
# year distribution

_ = plt.hist(df['year'], bins=20)

plt.show()
# mileage distribution

_ = plt.hist(df['mileage'], bins=20)

plt.show()
# relation between price and engine volume

_ = plt.scatter(df.price, df.engine)

plt.show()
# remove outliner with extremaly high engine volume

df = df[df.engine < 15000]
# relation between price and engine volume

_ = plt.scatter(df.price, df.engine)

plt.show()
# How much our DataFrame was reduced

len(df)
# how fast car prices go down with each year of usage?



# mean price for each year

unique_years = list(set(df.year.values))

mean_prices = []

for year in unique_years:

    temp_df = df[df["year"] == year]

    mean_prices.append(int(temp_df.price.mean()))



mean_prices = np.array(mean_prices)

_ = plt.scatter(mean_prices, unique_years)

plt.show()

_ = plt.plot(mean_prices, unique_years)

plt.show()