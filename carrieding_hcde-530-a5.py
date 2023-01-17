# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# assign the input a variable name, TPEListing

TPEListing = pd.read_csv("/kaggle/input/airbnbtpelisting/listings.csv")
# import pandaa data reader, matplotlib

import pandas_datareader.data as web

import matplotlib.pyplot as plt

import datetime as dt
# sort the value by minimum nights from small to large

# plot TPEListing in scatter with minimum nights as x-axis and price as y-axis

# price goes down as the required minimum nights increases.

TPEListing.sort_values(by = 'minimum_nights', ascending = True)

TPEListing.plot(kind='scatter', x = 'minimum_nights', y = 'price')
# plot TPEListing in scatter with availability 365 as x-axis and number of reviews as y-axis

# the more the reviews the less available the Airbnb, but it's very loosely coorelated.

TPEListing.plot(kind='scatter', x = 'availability_365', y = 'number_of_reviews', alpha=0.5)
# plot TPEListing in scatter with price as x-axis and availability 365 as y-axis

# As the price goes up, it's either very popular or vacant all the time

TPEListing.plot(kind='scatter', x = 'price', y = 'availability_365', alpha=0.5)