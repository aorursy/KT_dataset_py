import pandas as pd

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
data = pd.read_csv("../input/7210_1.csv", low_memory=False)

data = data[["colors", "prices.amountMin", "prices.amountMax"]]

# Getting rid of missing data

data.dropna(inplace=True)

data["prices.average"] = (data["prices.amountMin"]+data["prices.amountMax"])/2

data
data.colors.value_counts().head()
black_shoes = data[:][data["colors"] == "Black"]

brown_shoes = data[:][data["colors"] == "Brown"]

white_shoes = data[:][data["colors"] == "White"]

blue_shoes = data[:][data["colors"] == "Blue"]

silver_shoes = data[:][data["colors"] == "Silver"]

pink_shoes = data[:][data["colors"] == "Pink"]

other_shoes = data[:][data["colors"] != "Pink"]



ttest_ind(pink_shoes["prices.average"], other_shoes["prices.average"], equal_var=False)
plt.hist(other_shoes["prices.average"], bins=100, range=(0,500))

plt.title("Histogram of not pink shoes average price")
plt.hist(pink_shoes["prices.average"], bins=100, color="pink", range=(0,500))

plt.title("Histogram of pink shoes average price")
plt.hist(black_shoes["prices.average"], bins=100, color="black", range=(0,500))

plt.title("Histogram of black shoes average price")
plt.hist(brown_shoes["prices.average"], bins=100, color="brown", range=(0,500))

plt.title("Histogram of brown shoes average price")
plt.hist(white_shoes["prices.average"], bins=100, color="lightgrey", range=(0,500))

plt.title("Histogram of white shoes average price")
plt.hist(blue_shoes["prices.average"], bins=100, color="blue", range=(0,500))

plt.title("Histogram of blue shoes average price")
plt.hist(silver_shoes["prices.average"], bins=100, color="silver", range=(0,500))

plt.title("Histogram of silver shoes average price")
plt.hist(black_shoes["prices.average"], bins=100, color="black", range=(0,500), normed=True)

plt.hist(brown_shoes["prices.average"], bins=100, color="brown", range=(0,500), normed=True, alpha=0.75)

plt.hist(white_shoes["prices.average"], bins=100, color="green", range=(0,500), normed=True, alpha=0.5)

plt.hist(blue_shoes["prices.average"], bins=100, color="blue", range=(0,500), normed=True, alpha=0.5)

plt.hist(silver_shoes["prices.average"], bins=100, color="silver", range=(0,500), normed=True, alpha=0.5)

plt.hist(pink_shoes["prices.average"], bins=100, color="pink", range=(0,500), normed=True, alpha=0.5)

plt.legend(['Black', "Brown", "White", "Blue", "Silver", "Pink"])