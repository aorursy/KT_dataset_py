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
import matplotlib.pyplot as plt

%matplotlib inline
# Read data

df = pd.read_csv("../input/master.csv", dtype = 

                 {'country' : 'category', 'sex' : 'category', 

                  'age' : 'category', 'generation' : 'category'}).rename(columns = {'suicides/100k pop' : 'suicides_100k'})

df.head()
# method to calculate suicide_rate based on different aggregation (groupy criterias)

def calculate_suicide_rate(df1, groupby):

    df_suicide = (df1.groupby(groupby)['suicides_no', 'population'].

              agg('sum').

            rename(columns={

                            'suicides_no' : 'total_suicides',

                            'population' : 'total_population'

            }))

    

    ret_value = (df_suicide['total_suicides'] * 100)/ df_suicide['total_population']

    return ret_value
# Countries that have max suicide rates 



_ = calculate_suicide_rate(df, "country").sort_values(ascending=False).head(15).plot.bar()
# Countries that have min suicide rates

_ = calculate_suicide_rate(df, "country").sort_values(ascending=True).head(15).plot.bar()
# Check distribution of suicide rates across males/ females over the years across countries

# Across countries - the rate of suicides increased from 1985-> 1995, especially in males



_ = calculate_suicide_rate(df, ["year", "sex"]).unstack('sex').sort_index().plot()
# Countries where suicide rate for males have gone up betwen 1985 -> 1995



suicides_1985_per = calculate_suicide_rate(df.loc[(df.year == 1985) & (df.sex == 'male')], "country")

suicides_1995_per = calculate_suicide_rate(df.loc[(df.year == 1995) & (df.sex == 'male')], "country")



_ = (suicides_1995_per - suicides_1985_per).dropna().sort_values(ascending=False).head(15).plot.bar()
# Suicide rates by age group, 

# Suicide rates increases with age



_ = calculate_suicide_rate(df, "age").sort_values(ascending=True).plot.bar()
# Find countries where suicide rates have max standard devitation 



_ = calculate_suicide_rate(df, ["country", "year"]).unstack("year").std(axis=1).sort_values(ascending=False).head(10).plot.bar()
# Korea: the suicide rates have gone up with years



_ = calculate_suicide_rate(df.loc[df.country == 'Republic of Korea'], "year").sort_index().plot()
# Russia: Suicide rates have decreaed after 2000 



_ =calculate_suicide_rate(df.loc[df.country == 'Russian Federation'], "year").sort_index().plot()
# Is there a correlaton between per capita gdp and suicide rate for Russia

# As GDP increases from ~ 1999 - 2013, suicide rates have come down!



_ = (df.loc[df.country == 'Russian Federation'].groupby('year')['suicides_no', 'gdp_per_capita ($)'].

                             mean().

                             plot(y=['suicides_no', 'gdp_per_capita ($)'])

    )