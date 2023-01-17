# Crude Oil is one of the most important natural resource.
# Hence careful utilization of this resource is important. The below analysis helps us understand the variation of crude oil prices
# on a daily, weekly, monthly and yearly basis.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Reading the datasets

daily = pd.read_csv("../input/brent-crude-and-wti-oil-prices/data/brent-daily.csv")
weekly = pd.read_csv("../input/brent-crude-and-wti-oil-prices/data/brent-weekly.csv")
monthly = pd.read_csv("../input/brent-crude-and-wti-oil-prices/data/brent-monthly.csv")
yearly = pd.read_csv("../input/brent-crude-and-wti-oil-prices/data/brent-year.csv")
# Understanding the variation of daily prices vs weekly prices
import seaborn as sns
sns.scatterplot(x = daily['Price'],y=weekly['Price'])
#Distribution Plot
sns.distplot(a = daily['Price'],kde = False)
#kernel Density Estimate showing probability density of weekly price
sns.kdeplot(data = weekly['Price'])
#Line Plot showing monthly variation of crude oil prices
sns.lineplot(data = monthly['Price'])
#Line Plot depicting yearly variation of prices
sns.lineplot(data = yearly['Price'])
# Analysis of monthly vs yearly prices( x axis = Monthly variation y axis = Yearly Variation)
sns.swarmplot(x = monthly['Price'],y = yearly['Price'])