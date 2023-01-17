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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

df=pd.read_csv('/kaggle/input/canberra-real-estate-sales-20062019/property_sales_canberra.csv')

df.tail(10)
df['datesold'] = pd.to_datetime(df['datesold'])

df = df[np.abs(df.price - df.price.mean()) <= (5.0 * df.price.std())] # Clean the outliers

%matplotlib inline 

plt.figure(figsize=(15,5))

plt.plot_date(df['datesold'], df['price'], xdate=True, markersize=1)
# We will group and visualise the data by property type

import seaborn

from  matplotlib import pyplot

_propertytype=df['propertyType'].unique().sort()

fg = seaborn.FacetGrid(data=df, hue='propertyType', hue_order=_propertytype, aspect=2, height=8)

fg.map(pyplot.scatter, 'datesold', 'price', alpha=.7, s=5).add_legend()