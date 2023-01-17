# Now let's learn how to make a histogram using Python! *sssssss* 



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load economic data

shoes = pd.read_csv("../input/7210_1.csv")



# I tried to do it from memory and forgot about the pd. prefix. Pfft. 



# Let's check it out (akin to R's summary() function)

shoes.describe()
# Now for some histogramming. First, import visualization library

import matplotlib.pyplot as plt



# Funny, I first typed "matlabplot" because I'm doing something in Matlab at the moment too



# Let's get that first column. Ew, clunky compared to $ but it also fits df[x] syntax, sort of

prices = shoes["prices.amountMax"]



# Would .describe() work again here? Yes! 

prices.describe()
# Now to make the histogram

hist(prices)



# Well that didn't work. 

# Then I remembered I imported matplotlib.pyplot as plt so maybe I need plt?



plt.hist(prices)
# Not a terrific histogram so let's adjust bin sizes

plt.hist(prices, bins = 50) # bins ask for number of bins (or bars)
# Let's look at shoe prices < $1,000

plt.hist(prices, range = (0,1000), bins = 100)
# Oh cool, an interesting pattern - you can see there are shoes sold at $399, $499, $599, etc.

# Add title

plt.title("Histogram of Women's Max Shoe Prices < $1,000")
# Boo, no histogram. So I need both in the same call (or...code block? What do you call it?). 

plt.hist(prices, range = (0,1000), bins = 100)

plt.title("Histogram of Women's Max Shoe Prices < $1,000")



# Presto! 