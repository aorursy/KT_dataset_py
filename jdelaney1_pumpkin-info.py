# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import pylab

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
philly = pd.read_csv('../input/philadelphia_9-24-2016_9-30-2017.csv')



philly.head()
philly.describe()
philly.fillna('O')
types = philly[['Variety','Date', 'Package', 'High Price', 'Origin', 'Item Size' ]]



types.head()
types['Variety'].describe()
types.dtypes
types_group = types.groupby('Variety')

types_group.size()
types_totals = types_group.sum()

variety_plot = types_totals.plot(kind='bar')
origin_group = types.groupby('Origin')

origin_totals = origin_group.sum()



origin_plot = origin_totals.plot(kind='bar')
size_group = types.groupby('Item Size')

size_totals = size_group.head()



size_plot = size_totals.plot(kind='bar')
kind_group = types.groupby(['Date', 'Item Size']).sum()

kind_group.head()
kind_group.unstack().head()
my_plot = kind_group.unstack().plot(kind='bar', stacked=True, title='Pumpkin by Date and price')

my_plot.set_xlabel("Date Purchased")

my_plot.set_ylabel("Money spent")

pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.5))

import glob

glob.glob("../input/*.csv")
all_data = pd.DataFrame()



for f in glob.glob("../input/*.csv"):

    df = pd.read_csv(f)

    all_data = all_data.append(df,ignore_index=True)
all_data.describe()
extract_data = all_data[['Variety','Date', 'Package', 'High Price', 'Origin', 'Item Size', 'City Name' ]]



extract_data.describe()
cleanup_bins = {"Package": {"24 inch bins":24, "36 inch bins":36}}
extract_data.replace(cleanup_bins, inplace=True)

extract_data.head()

cleanup_items = {"Item Size": {"xlge":4, "lge":3, "med-lge":2.5, "med":2, "sml":1}}

extract_data.replace(cleanup_items, inplace=True)

extract_data.head()

extract_data.describe()
extract_data.dtypes
extract_data = extract_data.convert_objects(convert_numeric=True)



extract_data.dtypes
extract_data.describe()
state_data = extract_data[['Origin', 'Package', 'Item Size']]



state_group = state_data.groupby(['Origin', 'Package']).sum()



my_plot = state_group.unstack().plot(kind='bar', stacked=True, title='States and Pumpkin Production')

my_plot.set_xlabel("Origin")

my_plot.set_ylabel("Quantity")

pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.5))

price_data = extract_data[['City Name', 'High Price']]



price_group = price_data.groupby(['City Name']).mean()



my_plot = price_group.unstack().plot(kind='bar', stacked=True, title='City and Prices of Pumpkins')

my_plot.set_xlabel("City Name")

my_plot.set_ylabel("Price of Pumpkins")



size_data = all_data[['Item Size', 'High Price']]

size_group = size_data.groupby(['Item Size']).mean()



my_plot = size_group.unstack().plot(kind='bar', stacked=True, title='Pumpkin size and Price')

my_plot.set_xlabel("Pumpkin sizes")

my_plot.set_ylabel("Price of Pumpkins")





size_data = extract_data[['Variety', 'High Price']]

size_group = size_data.groupby(['Variety']).mean()



my_plot = size_group.unstack().plot(kind='bar', stacked=True, title='Pumpkin type and Price')

my_plot.set_xlabel("Pumpkin Types")

my_plot.set_ylabel("Price of Pumpkins")
