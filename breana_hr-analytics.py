# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set(style='white')



hr_data=pd.read_csv('../input/HR_comma_sep.csv')

hr_data.head()
corr=hr_data.corr()



mask=np.zeros_like(corr)

mask[np.triu_indices_from(mask)]=True



f,ax = plt.subplots(figsize=(9,9))

cmap=sns.diverging_palette(240, 14, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, ax=ax, square=True)

plt.title("Correlation Matrix")

plt.show()
left=hr_data.loc[hr_data['left']==1]



fig=plt.figure(figsize=(11,9))

ax1= fig.add_subplot(2,2,1)

ax2=fig.add_subplot(2,2,2)

ax3=fig.add_subplot(2,2,3)

ax4=fig.add_subplot(2,2,4)



sns.distplot(left['satisfaction_level'], ax=ax1)

sns.distplot(left['last_evaluation'], ax=ax2)

sns.distplot(left['average_montly_hours'], ax=ax3)

sns.distplot(left['time_spend_company'], ax=ax4)



ax1.set_title('Satisfaction Level Distribution')

ax1.set_xlabel('Satisfaction Level')

ax2.set_title('Last Evaluation Distribution')

ax2.set_xlabel('Last Evaluation')

ax3.set_title('Average Monthly Hours Distribution')

ax3.set_xlabel('Average Monthly Hours')

ax4.set_title('Time at Company Distribution')

ax4.set_xlabel('Time Spent at Company')

plt.tight_layout()

plt.show()
sales=hr_data.groupby('sales').sum()

sales
sales=hr_data.groupby('sales').mean()

sales
good_people_left=hr_data.loc[(hr_data['last_evaluation'] >= 0.7) & (hr_data['time_spend_company'] >= 4)]

corr=good_people_left.corr()



mask=np.zeros_like(corr)

mask[np.triu_indices_from(mask)]=True



f,ax = plt.subplots(figsize=(9,9))

cmap=sns.diverging_palette(240, 14, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap,  ax=ax, square=True)

plt.title("Correlation Matrix")

plt.show()

print(corr)