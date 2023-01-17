import numpy as np 

import pandas as pd 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
who_suicide_statistics = pd.read_csv("../input/who-suicide-statistics/who_suicide_statistics.csv")
uk_data = who_suicide_statistics[who_suicide_statistics['country'] == 'United Kingdom'];uk_data.head()
uk_data.info()
# drop rows with missing values

print(uk_data.shape)

uk_data = uk_data.dropna()

print(uk_data.shape)
%matplotlib inline

import matplotlib.pyplot as plt

uk_data.hist(bins=30, figsize = (20,15))

plt.show()
uk_data.groupby('age').suicides_no.sum().nlargest(10).plot(kind='barh')
uk_data.groupby('sex').suicides_no.sum().nlargest(10).plot(kind='barh')
uk_data.groupby(['age','sex']).suicides_no.sum().nlargest(10).plot(kind='barh')
uk_data.groupby('year').suicides_no.sum().plot()
ax = sns.catplot(x="sex", y="suicides_no",col='age', data=uk_data, estimator=np.median,height=4, aspect=.7,kind='bar')
ax = sns.catplot(x="sex", y="suicides_no",col='age', data=uk_data[uk_data['year'] == 2015], estimator=np.median,height=4, aspect=.7,kind='bar')
uk_data_2015 =uk_data[uk_data['year'] == 2015]; uk_data_2015
for name, group in uk_data_2015.groupby('age'):

    female = group[group['sex'] == 'female'].suicides_no

    male = group[group['sex'] == 'male'].suicides_no

    print(name + " male to female ratio is " + str(round(int(male)/int(female), 1)))