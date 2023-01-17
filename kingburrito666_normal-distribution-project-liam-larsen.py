import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/Computers.csv")

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
prices = df.price # Extracting the prices column

prices.head(10) # Printing the first 10 values

fig, ax = plt.subplots(figsize=(15, 7))

plt.xticks(rotation=45)

plt.yticks()

plt.style.use('ggplot')

plt.title("Computer Prices in the Early 90's")

plt.xlabel("Number")

plt.ylabel("Price")

ax.plot(prices.head(80))

plt.show()
fig = plt.subplots(figsize=(15, 10))

plt.xticks()

plt.yticks()

plt.title("Computer Prices in the Early 90's")

plt.xlabel("Number")

plt.ylabel("Price")

plt.boxplot(prices.head(300), 0, 'rs', 0)

plt.show()
# How to calculate mean, sum of all values / number of values

mean = prices.sum() / prices.size  



# Print values

print("Mean of data: " + str(mean) )

print("Mean of data using built in function: " + str(prices.mean()) )
# Lets plot a histogram to see if our data is normally distributed

fig= plt.subplots(figsize=(15, 10))

plt.xticks()

plt.yticks()

plt.title("Histogram of data")

plt.xlabel("Price")

plt.ylabel("Number")

plt.hist(prices)

plt.show()