# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
avocado_file_path = '.../input/avocado.csv'
avocado_data = pd.read_csv('../input/avocado.csv', index_col=0)

print(avocado_data.columns)
avocado_data.head()
avocado_data
reviews = avocado_data.loc[:,['year', 'Total Bags', 'AveragePrice', 'Total Volume']]
reviews.dropna()
reviews.describe
reviews.plot.scatter(x='year', y = 'AveragePrice')
reviews.plot.scatter(x='year', y = 'Total Bags')
reviews.plot.scatter(x='year', y = 'Total Volume')
# This shows the average price is at an all time low in the year 2018
# compared to the other years recorded so far.
# The second graph shows an increase in the volume of 
# product sold in the year 2018 compared to the others years. 
# There is a steady increase every year and this data correlates to the third 
# graph showing a simillar trend. 
# Conclusion: As the average price drops, the more product is sold.  
avocado_data.groupby('region').AveragePrice.max()
avocado_data.groupby('region').AveragePrice.max().plot.bar()
avocado_data.groupby('year').groups.keys()
avocado_data.groupby('type').groups.keys()
avocado_data.groupby('AveragePrice').max()
avocado_data.groupby('year').plot.bar()