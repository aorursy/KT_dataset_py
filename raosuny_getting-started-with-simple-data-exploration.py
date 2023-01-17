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
df = pd.read_csv('/kaggle/input/e-commerce-purchase-dataset/purchase_data_exe.csv')
df.head()
df.drop(['Unnamed: 7'], axis=1, inplace=True)
df.head()
df['payment_method'] = df['payment_method'].astype('category').cat.codes

#Now 0 represents credit and 1 represents paypal
df['payment_method'] = df['payment_method'].astype('int')
df.dtypes
from datetime import datetime



#df['DateTime'] = pd.to_datetime(df['date'])

df['Year']=[d.split('/')[2] for d in df.date]

df['Month']=[d.split('/')[1] for d in df.date]

df['Day']=[d.split('/')[0] for d in df.date]

df.Year = df.Year.astype('int')

df.Month = df.Month.astype('int')

df.Day = df.Day.astype('int')

df.drop(['date'], axis = 1, inplace= True)
df.head()
df.dtypes
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
plt.hist(df.product_category, bins = 20)

plt.xlabel("Product Category")

plt.ylabel("Frequency")

plt.show()
plt.hist(df.payment_method, bins = 2)

plt.xlabel("Payment Method")

plt.ylabel("Frequency")

plt.show()
plt.scatter(x = df["clicks_in_site"], y = df["value [USD]"])

plt.xlabel("Clicks in site")

plt.ylabel("Value in USD")

plt.show()
plt.scatter(x = df["time_on_site [Minutes]"], y = df["value [USD]"])

plt.xlabel("Time on Site in Minutes")

plt.ylabel("Value in USD")

plt.show()
plt.scatter(x = df["payment_method"], y = df["value [USD]"])

plt.xlabel("Payment Method")

plt.ylabel("Value in USD")

plt.show()
new=df.groupby("payment_method")["value [USD]"]

new.mean()
plt.scatter(x = df["product_category"], y = df["value [USD]"])

plt.xlabel("Product Category")

plt.ylabel("Value in USD")

plt.show()
new=df.groupby("product_category")["value [USD]"]

new.mean()
new=df.groupby("Day")["value [USD]"]

new.sum()
df.corr()