import matplotlib.pyplot as plt

import pandas as pd
#assign camera list data to variable dataset and describe it

dataset = pd.read_csv("../input/camera_dataset.csv")

dataset.describe()
# list all the coulmn names

print(dataset.columns)
# get the Price column

price = dataset["Price"]



# Plot a histogram of price content

plt.hist(price)

plt.title("price in Camera dataset")
# Plot a histogram of prices with 10 bins, a black edge 

# around the columns & at a larger size

plt.hist(price, bins=10, edgecolor = "black")

plt.title("Prices distribution in Camera dataset") # add a title

plt.xlabel("Prices") # label the x axes 

plt.ylabel("Count") # label the y axes
### another way of plotting a histogram (from the pandas plotting API)

# figsize is an argument to make it bigger

dataset.hist(column= "Price", figsize = (10,10))