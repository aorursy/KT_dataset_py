# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sal = pd.read_csv('/kaggle/input/sf-salaries/Salaries.csv')

sal.tail()
sal.info()
#lets drop the notes column as its empty  and we cant  fill it with anything

sal.drop("Notes",axis=1, inplace=True)
sns.set_palette('GnBu_r')

sns.set_style('whitegrid')

sns.distplot(sal['TotalPayBenefits'],bins=30);
sal['BasePay'].mean()
max_amount = sal['OvertimePay'].max()

max_amount
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['JobTitle']
sal[sal['EmployeeName'] == 'JOSEPH DRISCOLL']['TotalPayBenefits']
max_pay = sal['TotalPayBenefits'].max()

sal[sal['TotalPayBenefits'] == max_pay]['EmployeeName']
sal[sal['TotalPayBenefits'] == sal['TotalPayBenefits'].min()]['EmployeeName']
by_year = sal.groupby('Year')['BasePay']

by_year.mean()
sal['JobTitle'].nunique()
sal['JobTitle'].value_counts().head()
sum(sal[sal['Year']==2013]['JobTitle'].value_counts() == 1)
def check_title(title):

    if 'chief' in title.lower():

        return True

    else:

        return False
sum(sal['JobTitle'].apply(check_title))
title_len = sal['JobTitle'].apply(len)

peso = sal['TotalPayBenefits']
new_df = pd.DataFrame({'title_len':title_len,

                        'TotalPayBenefits':peso

                      })

new_df.head()
new_df.corr()