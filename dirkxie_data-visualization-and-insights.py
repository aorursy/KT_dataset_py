# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Press "shift + Enter" to run the cell in .ipynb file



import matplotlib.pyplot as plt # 2D plotting library

import seaborn as sns # a high-level plotting library

import warnings

warnings.simplefilter(action='ignore') # stop warnings 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data2 = pd.read_csv("../input/cleaned-data/Final_Data.csv")
# The distribution of total number of firms in town

sns.distplot(data2.total_firms)
data2.total_firms.quantile(0.9)
target = data2[data2.total_firms < data2.total_firms.quantile(0.9)]

sns.distplot(target.total_firms,axlabel='Total Number of Firms')
top10_firms = data2.nlargest(5,'total_firms')

top10_firms = top10_firms.reset_index()

top10_firms[['Town','total_firms']]
top10_firms_type = top10_firms.iloc[:,-4:]

top10_firms_type.columns = columns=['Micro','Small','Medium','Large']

top10_firms_type.index = index=top10_firms.Town

top10_firms_type.plot.bar(rot=0)
a = 0

for i in range(len(data2)):

    if data2.iloc[i,-4]<= (max(data2.iloc[i,-3],data2.iloc[i,-2],data2.iloc[i,-1])):

        a += 1

        print('Other scale company is equal or larger than Microcompany in',data2['Town'][i])

print('The number of cities that micro scale firms is not main kind of company: ',a)
data2.iloc[:,[2,-4,-3,-2,-1]][data2.Town.isin(

    ['Brenouille','Bihorel','Le Plessis-Pâté'])]
sns.distplot(data2.total_population)
target2 = data2[data2.total_population < data2.total_population.quantile(0.9)]

sns.distplot(target2.total_population,axlabel='Total Population')
top10_population = data2.nlargest(5,'total_population')

top10_population = top10_population.reset_index()

top10_population[['Town','total_population']]
data2['total_population'].corr(data2['total_firms'])
data2.mean_salary.describe()
sns.barplot(x=['mean_female_salary','mean_male_salary'],y=[data2.mean_female_salary.mean(),data2.mean_male_salary.mean()])
age_groups = pd.DataFrame([[data2.mean_young_female_salary.mean(),data2.mean_young_male_salary.mean()],

                   [data2.mean_medium_female_salary.mean(),data2.mean_medium_male_salary.mean()],

                   [data2.mean_old_female_salary.mean(),data2.mean_old_male_salary.mean()]],

                  columns=['female','male'],index=['young','medium','old'])

age_groups.plot.bar(rot=0)
job_category = pd.DataFrame([[data2.mean_female_employee_salary.mean(),data2.mean_male_employee_salary.mean()],

                             [data2.mean_female_worker_salary.mean(),data2.mean_male_worker_salary.mean()],

                            [data2.mean_female_middle_manager_salary.mean(),data2.mean_male_middle_manager_salary.mean()],

                            [data2.mean_female_executive_salary.mean(),data2.mean_male_executive_salary.mean()]],

                            columns=['female','male'],index=['employee','worker','middle_manager','executive'])

job_category.plot.bar(rot=0)
from IPython.display import Image

Image("../input/map-image/Map.PNG")
Target_fratures = data2[['mean_salary','total_firms','total_population','mean_executive_salary', 'mean_middle_manager_salary',

       'mean_employee_salary', 'mean_worker_salary','mean_male_salary','mean_female_salary',

                 'mean_young_age_salary','mean_medium_age_salary', 'mean_old_age_salary']]

corr = Target_fratures.corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr, annot=True)

plt.show()