# Python uses packages to simplify data manipulation and analysis. Three of the most important packages

# are called Numpy, Pandas, and Matplotlib

import numpy as np # linear algebra and data processing

import pandas as pd # data processing and file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for ploting data

import seaborn as sns # makes even nicer plots (might use, don't know)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
wine_data = pd.read_csv("/kaggle/input/wine-reviews/winemag-data-130k-v2.csv")

wine_data.head()
mask = wine_data [wine_data['price']>0]

prices = mask ['price']

prices.head()

scores = wine_data['points']
print ("Stats for Prices")

print ("Mean:",int(prices.mean()))

print ("Median:",prices.median())

print ("Mode:",prices.mode())

print ("Stats for Points")

print ("Mean:",int(scores.mean()))

print ("Median:",scores.median())

print ("Mode:",scores.mode())

print("variance of point scores: ", scores.var(ddof=0)) # normalizing by N-0 (just N) since this is population parameter

print("standard deviation of point scores: ", scores.std(ddof=0)) # by default N-1 would be used

print("variance of prices: ", prices.var(ddof=0)) 

print("standard deviation of prices: ", prices.std(ddof=0))
plt.figure (figsize = (20,5))

plt.title ('Histogram of Prices',size=25)

plt.xlabel ('Prices',size=15)

plt.ylabel ('# of Occurance',size=15)

plt.xticks(size=10)

plt.yticks(size=10)

val = int ((prices.max())/2)

prices.hist(bins= val)

plt.show()
plt.figure (figsize = (15,5))

plt.title ('Histogram of Scores',size=25)

plt.xlabel ('Scores',size=15)

plt.ylabel ('# of Occurance',size=15)

plt.xticks(size=10)

plt.yticks(size=10)

val1 = int(100/5)

scores.hist(bins=val1)

plt.show()