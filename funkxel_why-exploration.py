# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
data['satisfaction_level'].describe()
data.boxplot(['satisfaction_level','last_evaluation'], 'left')
data['time_spend_company'].value_counts()
data.left.value_counts()
data['satisfaction_level'].plot.kde()
data['last_evaluation'].plot.kde()
data.groupby('left')['last_evaluation'].plot.kde(legend=True)
data.groupby('left')['satisfaction_level'].plot.kde(legend=True)
import matplotlib.pyplot as plt

# inspired by: https://github.com/justmarkham/DAT4/blob/master/notebooks/08_linear_regression.ipynb

# fig, axs = plt.subplots(1, 2, sharey=True)

# data.plot(kind='scatter', x='number_project', y='average_montly_hours')

# data.plot(kind='scatter', x='number_project', y='time_spend_company', ax=axs[1])

data.boxplot(['average_montly_hours'],'number_project')
data.boxplot(['satisfaction_level'],'number_project')
data.boxplot(['last_evaluation'],'number_project')
import statsmodels.formula.api as smf



lm = smf.ols(formula='satisfaction_level ~ number_project + average_montly_hours + salary', data=data).fit()



lm.params
lm.summary()
import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 2, sharey=True)

data.groupby('left').plot(kind='scatter', x='satisfaction_level', y='last_evaluation')

# data.plot(kind='scatter', x='number_project', y='time_spend_company', ax=axs[1])
# data.satisfaction_level[1000:1100].plot()

import seaborn as sns



sns.distplot(data.satisfaction_level, rug=True)
data.satisfaction_level.describe()
sns.distplot(data.satisfaction_level[data.satisfaction_level<0.2])
plt.hist(data.satisfaction_level, bins=[0.0, 0.1, 0.20, 0.30, 0.40, 0.50, 0.60, 0.7,0.8,0.9,1.0])
sns.distplot(data.satisfaction_level, bins=10, rug=True)
sns.distplot(data.last_evaluation[data.left==1], bins=10)
sns.distplot(data.satisfaction_level[data.left==1], bins=10)
sns.distplot(data.average_montly_hours[data.left==1], bins=10)