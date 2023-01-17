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
import pandas as pd

data = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
data.info()
# Does sex of a person plays a role?

import matplotlib.pyplot as plt

alpha = data.loc[data.sex == 'male']['suicides/100k pop']

beta = data.loc[data.sex == 'female']['suicides/100k pop']

plt.hist([alpha, beta], label = ['Male','Female'], bins = 50)

plt.legend(loc='upper right')

plt.xlabel('suicides/100k pop')

print('mean_suicide-rate_males = ' + str(alpha.mean()))

print('mean_suicide-rate_females = ' + str(beta.mean()))

# Higher suicide rates among males
data.groupby('sex')['country'].agg('count')

plt.boxplot([alpha,beta], labels = ['Male','Female'])

plt.ylabel('suicides/100k pop')

# Males have more extreme values for suicide rates,It is mostly true in third world countries where males are the bread winner 

# & woman takes care of the household, it is the male who is more probable to harbor guilt & self doubt 

# due to lack of work & failures as females are almost always occupied with responsibility of raising children & managing household.

# Nevertheless everyone should be treated with kindness, we never know what they're going through in life.
#Lets study if there is any effect of age on suicide rates

data.boxplot(by="age", column="suicides/100k pop")

# A general trend, older people have higher rates possibly because of loneliness that comes with age.
data.groupby('generation')['country'].agg('count')
# Mean is a better measure to study the effect of generation on suicide rates as we dont have equal samples

# of each generation. 

data.groupby('generation')['suicides/100k pop'].agg('mean').sort_values(ascending= False).plot(kind = 'bar')

# G.I. Generation(1901-1927) followed by Silent Generation(1928-1945) followed by Boomers(1946-1964)

# Result is in line with our previous finding that more people age more is the suicide rate

# We should keep elders of our family with us & should give them the love & care they deserve.
# Where are we over time?

data.groupby('year')['suicides/100k pop'].agg('mean').plot(kind = 'line')

data.groupby('year')['country'].agg('count')

# General trend is suicide rates are decreasing from 1995, we see a sudden rise in 2016 but

# that can be due to sampling error as observations are significantly less for that year.
# To capture the association of country's economy with suicide rate, gdp per capita is a better measure

# than only gdp.

eco_data = data.groupby('gdp_per_capita ($)',as_index=False)['suicides/100k pop'].agg('mean')

plt.scatter(eco_data['gdp_per_capita ($)'],eco_data['suicides/100k pop'])

plt.ylabel('suicides/100k pop')

plt.xlabel('gdp_per_capita ($)')

# There is no solid distinction but general trend is country with better economy have lesser suicide rates.

# Countries with low gdp_per_capita may or maynot have high suicide rate but developed economies have less for sure

# A better way to judge an economy is by PPP but it is not present in our data.
#HDI

hdi_data = data.groupby('HDI for year',as_index=False)['suicides/100k pop'].agg('mean')

plt.scatter(hdi_data['HDI for year'],hdi_data['suicides/100k pop'])

plt.ylabel('suicides/100k pop')

plt.xlabel('HDI for year')

# HDI doesn't seem to be a good measure to predict suicide rate
# Although we might have some intuition about which countries are most vulnerable

# Lets see some of them as of 2015.

data_2015 = data.loc[data.year == 2015]

data_2015.groupby('country')['suicides/100k pop'].agg('mean').sort_values(ascending= False)[:15].plot(kind = 'bar')

#This came as a surprise to me, common thinking is that African countries will top the list

#but we see a lot of european nations as well.
# According to our analysis Age/generation, Sex are major factors & economy of a country also has some

# effect on the suicide rates observed.