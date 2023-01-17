# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
color = sns.color_palette()

%matplotlib inline
# Let's load the data 

data = pd.read_csv('../input/all_data.csv')

data.head()
# Let's have an overview of all the columns in the dataframe

data.info()
# Convert timestamp column to proper date-time format

data['timestamp'] = pd.to_datetime(data['timestamp'])



# Drop the time for now and let's stick only with date

data['timestamp'] = data['timestamp'].apply(lambda x : pd.datetime.date(x))



data.head()
plt.figure(figsize=(10,5))

plt.scatter(range(len(data)), np.sort(data.total_addresses.values),alpha=0.5)

plt.xlabel('Index')

plt.ylabel('Total addresses')

plt.title('Addresses distribution ')

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(range(len(data)), np.sort(data.blocksize.values),alpha=0.5)

plt.xlabel('Index')

plt.ylabel('Blocksize')

plt.title('Blocksize distribution ')

plt.show()
blocksize_ulimit = 8000

data[data.blocksize > blocksize_ulimit]
new_blocksize = data['blocksize'].loc[data.blocksize < blocksize_ulimit]



plt.figure(figsize=(10,10))

sns.distplot(new_blocksize.values, kde=False, bins=50)

plt.xlabel('Blocksize')

plt.ylabel('Count')

plt.title('Blocksize distribution ')

plt.show()
df = data[['timestamp','price_USD']]

df.plot(x='timestamp', y='price_USD',figsize=(10,5),fontsize=14);