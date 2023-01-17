import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="id")



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
"What is the most expensive airbnb in NYC and what are the specs."
my_data.groupby(by="host_id").price.agg([max]).sort_values("max", ascending=False).head()
my_data.sort_values("price", ascending=False).head()
print("The total amount of owners", my_data.host_id.unique().size)

print("The total amount of buildings", my_data.size)

print("The relationship between owners and buildings, it means 20 buildings per owner", (my_data.size /my_data.host_id.unique().size))
plt.figure(figsize=(20,8))



# Create a plot

sns.barplot(x="neighbourhood_group", y="price", data=my_data, palette="Blues_d") # Your code here



# Check that a figure appears below

step_4.check()
value=1500



manhattan = my_data[(my_data.neighbourhood_group == "Manhattan") & (my_data.price < value)]

broklyn = my_data[(my_data.neighbourhood_group == "Broklyn") & (my_data.price < value)]

queens = my_data[(my_data.neighbourhood_group == "Queens") & (my_data.price < value)]

statenIsland = my_data[(my_data.neighbourhood_group == "Staten Island") & (my_data.price < value)]

bronx = my_data[(my_data.neighbourhood_group == "Bronx") & (my_data.price < value)]

plt.figure(figsize=(20,10))



sns.distplot(a=manhattan.price, label="Manhattan", kde=False) # Your code here

sns.distplot(a=broklyn.price, label="Broklyn", kde=False) # Your code here

sns.distplot(a=queens.price, label="Queens", kde=False) # Your code here

sns.distplot(a=statenIsland.price, label="Staten Island", kde=False) # Your code here

sns.distplot(a=bronx.price, label="Bronx", kde=False) # Your code here



# Add title

plt.title("Histogram of Airbnb price, by neighbourhood_group")



# Force legend to appear

plt.legend()
city_total = my_data.neighbourhood_group.value_counts()
plt.figure(figsize=(20,10))



sns.barplot(x=city_total.index, y=city_total, capsize=.2, palette="Blues_d")



plt.title("Graphic bar of Airbnb quantitier per city")

my_data.head()
plt.figure(figsize=(20,10))

sns.scatterplot(x='latitude', y='longitude', data=my_data, hue="neighbourhood_group")