# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
rates_data = pd.read_csv("../input/master.csv")

print(rates_data.columns)



# filter data for Slovenia

slo_rates_data = rates_data.loc[rates_data['country'].isin(['Slovenia'])]

print(slo_rates_data[:5])
import seaborn as sns

import matplotlib.pyplot as plt



# short suicides no. per year to get a sense of data

plt.figure(figsize=(10,6))

sns.barplot(x=slo_rates_data['year'], y=slo_rates_data['suicides_no'])

# short suicides no. per year to get a sense of data - also we have color coded sex

# it's interesting that there are a lot more male suicides than female according to data

sns.scatterplot(x=slo_rates_data['year'], y=slo_rates_data['suicides_no'], hue=slo_rates_data['sex'])
# further exploration into potential relationship between sex and suicides no

# there's a larger group of male suicides present pending further exploration

sns.swarmplot(x=slo_rates_data['sex'], y=slo_rates_data['suicides_no'])
# according to wiki boomers were born between 1946 and 1964 - since this dataset is between 1985 and 2016 estimate critical suicide group age

# is between 21 and 72 - basically adult male - it's interesting that the suicide no for females has a completely different structure

sns.swarmplot(x=slo_rates_data['sex'], y=slo_rates_data['suicides_no'], hue=slo_rates_data['generation'])