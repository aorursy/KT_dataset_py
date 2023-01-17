# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
data = pd.read_csv('../input/global-super-store-dataset/Global_Superstore2.csv', encoding='latin1')

data['Order Date'] = pd.to_datetime(data['Order Date'])

data = data.set_index('Order Date')

data = data.sort_values(by='Order Date', ascending=True)

data.head()
sector = data.groupby('Country').resample('A').sum()

sector = sector.sort_values(by='Profit', ascending=False)

#sector = sector.groupby('Category').resample('A').sum()

sector = sector['Profit'][:120]

sector.head()
product = data.groupby('Product Name').resample('A').sum()

product = product.sort_values(by='Quantity', ascending=False)

product = product['Quantity'][:50]

product
product.unstack().plot(kind='bar', stacked=True, figsize=(20,7))
sector.unstack().plot(kind='bar', stacked=False, figsize=(20,7))

plt.grid()
category = data.groupby('Sub-Category').resample('A').count()

category = category['Sub-Category']

category.head()
category.unstack().plot(kind='bar', stacked=False)

plt.title('Sales by category')
profit = data.groupby('Sub-Category').resample('A').sum()

profit = profit['Profit']

profit.head()
profit.unstack().plot(kind='bar', stacked=False)

plt.title('Profit by category')