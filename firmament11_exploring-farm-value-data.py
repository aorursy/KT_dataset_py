%matplotlib inline

import warnings

warnings.simplefilter('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from scipy.stats import norm, skew

import statsmodels.api as sm
ifva = pd.read_excel('../input/IOWAFarmData.xlsx')
ifva.head()
ifva = ifva.rename(index=str, columns={'Weighted Average value per acre': 'Farm Value'})

fig, ax = plt.subplots()

ax.scatter(x = ifva['Pasture Cash Rent'], y = ifva['Farm Value'])

plt.ylabel('Farm Value', fontsize=13)

plt.xlabel('Cash Rent', fontsize=13)

plt.show()
ifva['Farm Value'] = pd.to_numeric(ifva['Farm Value'], errors='coerce')

ifva = ifva.dropna(subset=['Farm Value'])

ifva['Farm Value'] = ifva['Farm Value'].astype(int)



sns.distplot(ifva['Farm Value'] , fit=norm);



(mu, sigma) = norm.fit(ifva['Farm Value'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')

            

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(ifva['Farm Value'], plot=plt)

plt.show()
Cols_to_Drop = ['Jan all grades cattle prices', 'Feb all grades cattle prices', 'Mar all grades cattle prices',

                'Apr all grades cattle prices', 'May all grades cattle prices', 'Jun all grades cattle prices',

                'Jul all grades cattle prices', 'Aug all grades cattle prices', 'Sep all grades cattle prices',

                'Oct all grades cattle prices', 'Nov all grades cattle prices', 'Dec all grades cattle prices',

                'High grade of land value per acre', 'Medium grade of land value per acre',

                'Low grade of land value per acre', 'CropLand Rent as % of value', 'Pasture Rent as % of value']

ifva.drop(columns = Cols_to_Drop, inplace = True)





#Farm Value Correlation Matrix

corrmat = ifva.corr()

plt.subplots(figsize=(12,9))

mask = np.zeros_like(corrmat, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corrmat, mask=mask, vmax=0.9, cmap="YlGnBu",

            square=True, cbar_kws={"shrink": .5})