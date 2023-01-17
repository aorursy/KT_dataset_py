# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Pull up the avocado



df = pd.read_csv('../input/avocado.csv')

df.head(10)
# Let's check if there is missing.

# Thanksfully no missing avocado

df.isna().sum()


# Seems 1 ~ 2.5 is fair price for avocado

plt.figure(figsize=(10,5))

plt.title("Price")

ax = sns.distplot(df["AveragePrice"], color = 'green')
# Let's see the price by type...

# Definitely organic one is more expensive than regular.

# 1.5 ~ 2.0 is fair price for orgnaic avocado, which I can't afford sadly... :(

conventional = df[df['type'] =='conventional']

organic = df[df['type'] =='organic']



plt.figure(figsize=(10,5))

plt.title("Conventional Price")

ax = sns.distplot(conventional["AveragePrice"], color = 'green')

plt.title("Organic ")

ax = sns.distplot(organic["AveragePrice"], color = 'orange')

# Let's see the price range by region.

# Conventional avocado price by region

g = sns.factorplot('AveragePrice','region',data=conventional,

                   hue='year',size=10,aspect=0.6,palette='Greens',join=False)
# Organic avocado price by region

# San Francisco is going fire! It's been almost 2.5, which was more than twice of regular avocado in 2017

g = sns.factorplot('AveragePrice','region',data=organic, hue='year',

                   size=10,aspect=0.6,palette='Oranges',join=False)
# Next, let's see whcih variable affects price more.

# But before that, I gotta change some data types.



typemap = {'conventional' : 0 , 'organic' : 1}

df['type'] = df['type'].map(typemap)

df.sample(10)




# I only put price,type,year,total volume, total bags on correlation analysis.

# Type definitely affects price most.



cols = ['AveragePrice','type','year','Total Volume','Total Bags']

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale = 1.0)

hm = sns.heatmap(cm,cbar = True, annot = True,square = True, 

                 fmt = '.2f', annot_kws = {'size':15}, yticklabels = cols, xticklabels = cols)
# This code is very import to conduct fbprophet.

# For some reason, fbprophet is not working in Kaggle without the code below.

# Don't forget turning on internet on the kernel.



!pip3 uninstall --yes fbprophet

!pip3 install fbprophet --no-cache-dir --no-binary :all:
# Setting up the column names as ds and y.

# column name must be 'ds' and 'y'



from fbprophet import Prophet

df =df[df['region'] =='TotalUS']

df1 = df[['Date','AveragePrice']]

df1['Date'] = df1['Date'].astype('datetime64[ns]')

df1 = df1.rename(columns = {'Date':'ds','AveragePrice':'y'})

df1.plot(x='ds',y='y',kind='line')



# The graph is averace price trend.


# Create price forecast of avocado.

# I sat the period as 365, which means 1 year forecast.



m = Prophet()

m.fit(df1)



future = m.make_future_dataframe(periods = 365)

forecast = m.predict(future)



# the graph shows avocado price likely goes up.



fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)
# Changepoints

# It doesn't really shows exact changepoints, so edit scale of changepoint 



from fbprophet.plot import add_changepoints_to_plot

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)

# looks much better. It closely shows changepoints



m = Prophet(changepoint_prior_scale=0.5)

forecast = m.fit(df1).predict(future)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)
# Uncertainty in the Trend



forecast = Prophet(interval_width=0.95).fit(df1).predict(future)

m = Prophet(mcmc_samples=300)

forecast = m.fit(df1).predict(future)

fig = m.plot_components(forecast)


# Diagnostics



from fbprophet.diagnostics import cross_validation

df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')

df_cv.tail()
from fbprophet.diagnostics import performance_metrics

df_p = performance_metrics(df_cv)

df_p.tail()
from fbprophet.plot import plot_cross_validation_metric

fig = plot_cross_validation_metric(df_cv, metric='mape')
# So how much would recentavocado price be?

# Average price of avocado would be $1.60

forecast.tail()