# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.stats as stats

import seaborn as sns

from matplotlib import rcParams



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/kc_house_data.csv')

df.head()

df.isnull().any()
df.dtypes
df.describe()
fig = plt.figure(figsize=(12, 6))

sqft = fig.add_subplot(121)

cost = fig.add_subplot(122)



sqft.hist(df.sqft_living, bins=80)

sqft.set_xlabel('Ft^2')

sqft.set_title("Histogram of House Square Footage")



cost.hist(df.price, bins=80)

cost.set_xlabel('Price ($)')

cost.set_title("Histogram of Housing Prices")



plt.show()
import statsmodels.api as sm

from statsmodels.formula.api import ols
m = ols('price ~ sqft_living', df).fit()

print(m.summary())
m = ols('price ~ sqft_living + bedrooms + grade + condition',df).fit()

print (m.summary())
sns.jointplot(x="sqft_living", y="price", data=df, kind = 'reg',fit_reg= True, size = 7)

plt.show()