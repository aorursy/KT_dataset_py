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
#Importing necessary libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns
House = pd.read_csv("../input/flat-price-vs-sqft/House.csv")
House.head()
House.describe()
House.shape
House.columns
sns.regplot(x="sqft", y="price", data=House);
sns.lmplot(x="sqft", y="price", data=House);
sns.lmplot(x="sqft", y="price", data=House, x_estimator=np.mean);
plt.hist(House.sqft)

plt.show
plt.hist(House.price)

plt.show
plt.boxplot(House.price)

plt.show
plt.boxplot(House.sqft)

plt.show
#Dropping the outlier rows with standard deviation

factor = 3

upper_lim = House['sqft'].mean () + House['sqft'].std () * factor

lower_lim = House['sqft'].mean () - House['sqft'].std () * factor



House2 = House[(House['sqft'] < upper_lim) & (House['sqft'] > lower_lim)]
House2.shape
#Dropping the outlier rows with standard deviation

factor = 3

upper_lim = House['price'].mean () + House['price'].std () * factor

lower_lim = House['price'].mean () - House['price'].std () * factor



House2 = House[(House['price'] < upper_lim) & (House['price'] > lower_lim)]
House2.shape
plt.boxplot(House2.sqft)               # Checking whether outliers is been removed

plt.show
plt.boxplot(House2.price)                 # Checking whether outliers is been removed

plt.show
House2.sqft.corr(House2.price)
import statsmodels.formula.api as smf
model = smf.ols("sqft ~ price", data = House2).fit()
model.params
model.summary()
model.conf_int(0.05)
pred= model.predict(House2)

pred
model2 = smf.ols("sqft ~np.log(price)", data = House2).fit()  # applying log function
model2.summary()
model2.params
model3 = smf.ols("np.log(sqft)~price", data = House2).fit() #applying exponential formula
model3.summary()
# quadratic model

House2["sqft_Sq"] = House2.sqft*House2.sqft

model_quad = smf.ols("price~sqft + sqft_Sq",data = House2).fit()
model_quad.params
model_quad.summary()
# To increse the r2 value we have applies all the log function including exponential formual & qudratic function the accuracy have been only increase for the log function