import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from plotly.offline import iplot

import plotly as py

import plotly.tools as tls

import cufflinks as cf

import pylab 

import scipy.stats as stats

py.offline.init_notebook_mode(connected=True)

cf.go_offline()
gross_en_data=pd.read_csv("../input/indian-school-education-statistics/gross-enrollment-ratio-2013-2016.csv")
gross_en_data.head()
gross_en_data.info()
gross_en_data.plot(kind='box',figsize=(20,10))
gross_en_data.plot.hist()
plt.style.available
plt.style.use('fast')
gross_en_data.plot.scatter(x='Primary_Boys',y='Primary_Girls',c='Primary_Total')

gross_en_data.plot.scatter(x='Upper_Primary_Boys',y='Upper_Primary_Girls',c='Upper_Primary_Total')

gross_en_data.plot.scatter(x='Secondary_Boys',y='Secondary_Girls',c='Secondary_Total')
gross_en_data.iplot()
gross_en_data.iplot(x='Primary_Total',y='Upper_Primary_Total',mode='markers+text',size=20)
gross_en_data.iplot(kind='box')
cf.getThemes()
cf.set_config_file(theme='space')
gross_en_data.iplot(kind='bar',barmode='stack',bargap=0.6)
plt.close();

sns.set_style("whitegrid");

sns.pairplot(gross_en_data)

plt.show()
# checking if any feature belong to Normal Distribution

plt.figure(figsize=(20,20))

plt.subplot(221)

stats.probplot(gross_en_data.Primary_Boys.values, dist="norm", plot=pylab)

plt.subplot(222)

stats.probplot(gross_en_data.Primary_Girls.values, dist="norm", plot=pylab)

plt.subplot(223)

stats.probplot(gross_en_data.Upper_Primary_Boys.values, dist="norm", plot=pylab)

plt.subplot(224)

stats.probplot(gross_en_data.Upper_Primary_Girls.values, dist="norm", plot=pylab)



pylab.show()
