import pandas as pd

import numpy as np

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
data = pd.read_excel('/kaggle/input/market-mix/market.xlsx')
data.head()
data.columns
data['BrandName'].unique()
len(data['BrandName'].unique())
data.groupby(['BrandName']).size().reset_index(name='counts')
# Selected a single brand to work on

Absolut_Vod = data[data['BrandName'] == 'Absolut']

Absolut_Vod.head()
Price_Absolut = Absolut_Vod[['LnSales','LnPrice']]
plt.scatter(Price_Absolut['LnPrice'],Price_Absolut['LnSales'])

plt.title('Normalized price vs sales')

plt.xlabel('Price')

plt.ylabel('Sales')

plt.show()
# Regression Plot using seaborn

import seaborn as sns; sns.set(color_codes=True)

plot = sns.regplot(x = Price_Absolut['LnPrice'],y = Price_Absolut['LnSales'], data=Price_Absolut)
import statsmodels.formula.api as sm

from statsmodels.compat import lzip

import numpy as np

import matplotlib.pyplot as plt

import statsmodels.graphics.regressionplots
reg_result = sm.ols(formula = 'LnSales ~ LnPrice',data = Price_Absolut).fit()
reg_result.summary()
name = sm.ols(formula = 'LnSales ~ LnPrice', data = Price_Absolut)

name.endog_names #dependent variable

name.exog_names #intercept and predictor
r = name.fit()
r.params
name.loglike(r.params)



# We can see the log likelihood ratio from the above summary stats and here
name.predict(r.params, [[1, 4.7]])



# In terms of linear regression, y = mx + c, c = 2.836674 and m = 1.130972, 

# we are passing two values x = 4.7 and 

# the the value 1 is passed as multiplier for c (so, c remains at 2.836674 as per our model)
# Statsmodels Regression Plots

fig = plt.figure(figsize=(15,8))

fig = statsmodels.graphics.regressionplots.plot_regress_exog(reg_result, "LnPrice", fig=fig)
#Let's add more indicators to the regression and to monitor the R-squared value, 

# our aim is to increase R-squared (or to determine the optimum level)

Additional_Absolut = Absolut_Vod[['LnSales','LnMag','LnNews','LnOut','LnBroad','LnPrint','LnPrice']]
result_2 = sm.ols('LnSales ~ LnMag + LnNews + LnOut + LnBroad + LnPrint + LnPrice',data=Additional_Absolut).fit()
result_2.summary()
# Statsmodels Multivariate Regression Plots

fig = plt.figure(figsize=(15,8))

fig = statsmodels.graphics.regressionplots.plot_partregress_grid(result_2, fig=fig)
interaction = sm.ols('LnSales ~ LnMag + LnNews + LnOut + LnBroad * LnPrint + LnPrice',data=Additional_Absolut).fit()
interaction.summary()
# Plots

fig = plt.figure(figsize=(15,8))

fig = statsmodels.graphics.regressionplots.plot_partregress_grid(interaction, fig=fig)