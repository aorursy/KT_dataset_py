# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
plt.style.use("fivethirtyeight")

df = pd.read_csv("../input/SY2016-2017_K-12_Socrata.csv")

df['num complete'] = df['COMPLETE']*.01*df['Total Enrollment'] #change percentages into numbers so that they can be charted against total enrollment

df
aggregation_functions = {'Total Enrollment': 'sum', 'num complete':'sum'} #add up total enrollment and num complete row data

df_new = df.groupby(df['Within District Boundary']).aggregate(aggregation_functions) #aggregate school data into their districts

df_new.sort_values(by='Total Enrollment', ascending=False) #sort by total enrollment
df_new.sort_values(by='Total Enrollment', ascending=False).plot(kind="bar", figsize=(12,8))
 #change percentages into numbers so that they can be totaled and compared

df['Medical Exemption Total'] = df['Medical Exemption']*.01*df['Total Enrollment']

df['Personal Exemption Total'] = df['Personal Exemption']*.01*df['Total Enrollment']

df['Religious Exemption Total'] = df['Religious Exemption']*.01*df['Total Enrollment']

df['Religious Membership Exemption Total'] = df['Religious Membership Exemption']*.01*df['Total Enrollment']

df
aggregation_functions = {'Medical Exemption Total': 'sum', 'Personal Exemption Total':'sum', 'Religious Exemption Total':'sum', 'Religious Membership Exemption Total':'sum'} #add up each column of exemption totals

df_totals = df.aggregate(aggregation_functions) #aggregate sums into new series

df_totals.plot(kind="pie", figsize=(12,12)).set_ylabel('Vaccination Exemption By Reason') #plot the totals data as a pie chart to show which reasons for exemptions are most used