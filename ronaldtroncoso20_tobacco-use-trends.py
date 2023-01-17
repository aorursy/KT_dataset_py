import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

import seaborn as sns

warnings.filterwarnings("ignore")
tb = pd.read_csv('../input/tobacco.csv')

print(tb.head())
tb.info()
columns = ['Smoke everyday', 'Smoke some days', 'Former smoker', 'Never smoked']



for x in columns:

    tb[x] = tb[x].str.strip('%').astype('float')

    
tb.head()
tb_group = tb.groupby(['Year'], as_index = False).mean()



fig = plt.figure(figsize = (8,6))

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)



tb_group.head()



y = 'Percentage of people'

x = 'Year'



ax1.set(title = 'Smoke everyday', ylabel = y, xlabel = x)

ax2.set(title = 'Smoke some days', ylabel = y, xlabel = x)

ax3.set(title = 'Former smoker', ylabel = y, xlabel = x)

ax4.set(title = 'Never smoked', ylabel = y, xlabel = x)

ax1.scatter(tb_group.Year, tb_group['Smoke everyday'], )

ax2.scatter(tb_group.Year, tb_group['Smoke some days'])

ax3.scatter(tb_group.Year, tb_group['Former smoker'])

ax4.scatter(tb_group.Year, tb_group['Never smoked'])



fig.tight_layout()

fig.autofmt_xdate()

plt.show()
from scipy import stats



states = set(tb.State)



slope_dict = {}



for state in states:

    slope, intercept, r_value, p_value, std_err = stats.linregress(tb.Year[tb.State == state], tb['Never smoked'][tb.State == state])

    slope_dict[state] = slope

    

slope_df = pd.DataFrame([slope_dict]).transpose()

slope_df.columns = ['slope']

slope_df.sort(columns = 'slope', ascending = True, inplace = True)
slope_dict1 = {}



for state in states:

    slope, intercept, r_value, p_value, std_err = stats.linregress(tb.Year[tb.State == state], tb['Smoke everyday'][tb.State == state])

    slope_dict1[state] = slope

    

slope_df1 = pd.DataFrame([slope_dict1]).transpose()

slope_df1.columns = ['slope']

slope_df1.sort(columns = 'slope',ascending = False, inplace = True)
slope_df.plot(kind = 'bar', figsize = (10,6), title = 'Never Smoked: % Changes from 1994 to 2010')

slope_df1.plot(kind = 'bar', figsize = (10,6), title = 'Smoke everyday: % Changes from 1994 to 2010')

plt.show()