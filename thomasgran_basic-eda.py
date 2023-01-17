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
vehicles = pd.read_csv("/kaggle/input/craigslist-carstrucks-data/vehicles.csv")
vehicles.head()
# Show the number of vehicles on the websites by year
vehicles[vehicles.year > 1990].year.value_counts().sort_index().plot()
import matplotlib.pyplot as plt
vehicles.manufacturer.value_counts().plot.bar(figsize = (15,10))
plt.title("Number of cars by manufacturer")
plt.show()
vehicles[vehicles.price < 60000].price.plot.box()
vehicles.state.value_counts().plot.bar(figsize = (15,10))
plt.title("Number of cars for each state")
plt.show()
#1