# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Nashville_housing_data_2013_2016.csv')
df.shape
df.columns
df[:5]
df.info()
df.describe()
df['Year'] = df['Sale Date'].str[:4]

df['Month'] = df['Sale Date'].str[5:7]

df['Date']= df['Sale Date'].str[8:10]
df['Property City'].value_counts().plot(kind='barh')
plt.hist(df['Sale Price'], bins=20)
df['log_sale_price'] = np.log(df['Sale Price'])

plt.hist(df['log_sale_price'], bins=20)
df['Land Use'].value_counts()[:10].plot(kind='barh')
# to be continued