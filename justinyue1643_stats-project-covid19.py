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
dataset = pd.read_csv('/kaggle/input/dataset/total_cases.csv')
data.head()
dataset.describe()
dataset.drop(columns = ['World'])
data = dataset.copy()
#The dataset has some missing values
#Our best solution was to impute the
#missing value with 0s for the time
#being
def clean(data):
    for r in range(1, len(data.columns)):
        country = data.columns[r]
        print(country)
        for c in range(len(data[country])):
            if(np.isnan(data[country][c])):
                data[country][c] = 0
    return data

data = clean(data)
import math
#Function for finding the z-value
def get_z_val(a1, sd1, a2, sd2, n1, n2):
    return (a1 - a2)/math.sqrt((sd1 ** 2)/n1 + (sd2 ** 2)/n2)
#gdp_above: sample of countries that are more economically developed
#gdp_below: sample of countries that are less economically developed
gdp_above = ['United States', 'China', 'Japan', 'Germany', 'India']
gdp_below = ['Brazil', 'Armenia', 'Zimbabwe', 'Madagascar', 'Iceland'] 
#snippet to be placed on the presentation
data[gdp_above].describe()
#snippet to be placed on the presentation
data[gdp_below].describe()
#We calculate the average amount 
#of confirmed cases for both samples. This function 
#does exactly that. 
def get_mean(countries):
    mean = 0
    for i in countries:
        mean += data[i].count() * data[i].mean()

    return (mean/(data['United States'].count()))
#We calculate the standard deviation
#average amount  of confirmed for both samples. 
#This function does exactly that. 
def get_std(countries):
    variance = 0
    for i in countries:
        variance += (data[i].std() ** 2)
    return variance ** (1/2)
#m1: mean of sample more economically developed
#s1: standard deviation of sample more economically developed
#m2: mean of sample less economically developed
#s2: standard deviation of sample less economically developed

m1 = get_mean(gdp_above)
s1 = get_std(gdp_above)
m2 = get_mean(gdp_below)
s2 = get_std(gdp_below)
get_z_val(m1, s1, m2, m1, data['United States'].count(), data['Brazil'].count())