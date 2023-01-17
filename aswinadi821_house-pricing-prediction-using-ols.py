import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from matplotlib import rcParams
%matplotlib inline
%pylab inline
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")

df.head()
df.isnull().any()
df.dtypes
df.describe()
fig=plt.figure(figsize=(12,6))

sqft=fig.add_subplot(121)

cost=fig.add_subplot(122)





sqft.hist(df.sqft_living, bins=80)

sqft.set_xlabel('Ft^2')

sqft.set_title('Histogram of house square footage')





cost.hist(df.price, bins=80)

sqft.set_xlabel('Price($)')

sqft.set_title('Histogram of house pricing')



plt.show()
import statsmodels.api as sm

from statsmodels.formula.api import ols
m=ols('price~sqft_living',df).fit()

print(m.summary())
m=ols('price~sqft_living+bedrooms+grade+condition',df).fit()

print(m.summary())
sns.jointplot(x='sqft_living',y='price',data=df, kind='reg', fit_reg=True, height=7)

plt.show()