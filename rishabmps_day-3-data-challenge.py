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
data = pd.read_csv("../input/7210_1.csv")
print(data.describe())
data.columns
data.reviews[0]
import matplotlib.pyplot as plt

%matplotlib inline
plt.hist(data['prices.amountMin'], bins=1000)

plt.title("Shoe Price Histogram")

plt.xlabel("Min Price")

plt.ylabel("Frequency")

plt.xlim(0,500)

plt.show()

print(data.isnull().sum())
data['price'] = (data['prices.amountMin']+data['prices.amountMax'])/2
data.price.plot.hist(bins=1000)

plt.title("Shoe Price Histogram")

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.xlim(0,300)

plt.show()

data['prices.color'].head()
data['colors'].head()
data.colors.value_counts()[1:10]
data_pink = data[data['colors']=='Pink']
data_not_pink = data[data['colors']!='Pink']
np.mean(data_pink['price'])
np.mean(data_not_pink['price'])
from scipy.stats import ttest_ind
ttest_ind(data.price,data_not_pink.price,equal_var=False)