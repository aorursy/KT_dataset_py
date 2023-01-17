# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



degree = pd.read_csv('../input/degrees-that-pay-back.csv')

salary = pd.read_csv('../input/salaries-by-college-type.csv')

region = pd.read_csv('../input/salaries-by-region.csv')
def change_to_num(money):

    if type(money)== str and money[0] == '$':

        a = money[1:].replace(',','')

        return float(a)

    else:

        return money

type(change_to_num('$234,54.00')) 
degree.sort_values('Mid-Career Median Salary', ascending=False).head(20)
salary.sort_values('Mid-Career Median Salary', ascending=False).head(10)
degree = degree.applymap(change_to_num)

salary = salary.applymap(change_to_num)

region = region.applymap(change_to_num)

sns.barplot(y = 'Undergraduate Major', x = 'Starting Median Salary', data = degree.sort_values('Starting Median Salary'))

fig = plt.gcf()

fig.set_size_inches(15, 20)

sns.barplot(y = 'Undergraduate Major', x = 'Percent change from Starting to Mid-Career Salary', data = degree.sort_values('Percent change from Starting to Mid-Career Salary', ascending= False))

fig = plt.gcf()

fig.set_size_inches(15, 20)
b = sns.barplot(y = 'Undergraduate Major', x = 'Mid-Career Median Salary', data = degree.sort_values('Mid-Career Median Salary'))

fig_1 = plt.gcf()

fig_1.set_size_inches(15, 20)

salary.groupby('School Type').mean().plot(kind = 'bar', figsize = (10,7))
region.groupby('Region').mean().plot(kind = 'bar', figsize = (10, 7))