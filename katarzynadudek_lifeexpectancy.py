# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
d = pd.read_csv("../input/life-expectancy-who/Life Expectancy Data.csv") #read data from csv file
d.describe()
list(d.columns.values)
df=d[['Country', 'Year','Life expectancy ', 'Adult Mortality','infant deaths', 'percentage expenditure', 'Hepatitis B','under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', 'Income composition of resources', ' HIV/AIDS']]
df.head()
df.describe()
df.mode()
import pandas_profiling
profile = pandas_profiling.ProfileReport(df)
profile.to_file("data_report_df.html")
first=d[['Life expectancy ', 'Adult Mortality','under-five deaths ']]

second=d[['Country', 'Year', 'Life expectancy ', 'percentage expenditure', 'Total expenditure']]


third=d[['Country','Life expectancy ', 'Hepatitis B','Polio','Diphtheria ']]

df.corr()
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(df.corr(), annot=True)
plt.show()
# Seaborn visualization library
import seaborn as sns# Create the default pairplot
sns.pairplot(second)
plt.savefig('./second.png')
plt.show()

#ax = second.plot.line(y='Life expectancy ',x='percentage expenditure', color='DarkBlue', label='Adult Mortality');

_ = sns.lmplot(y='Life expectancy ',x='percentage expenditure', data=second, ci=None)
#plt.savefig('./seond2.png')

sns.pairplot(first)
plt.savefig('./first.png')
plt.show()

ax = first.plot.scatter(y='Life expectancy ',x='Adult Mortality', color='DarkBlue', label='Adult Mortality');

first.plot.scatter(y='Life expectancy ',x='under-five deaths ', color='DarkGreen', label='Under-five Deaths', ax=ax);

plt.savefig('./first2.png')



sns.pairplot(third)
plt.savefig('./third.png')
plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(third, alpha=0.2, figsize=(15, 15), diagonal='hist')
plt.savefig('./third2.png')

plt.show()
#'Country','Life expectancy ', 'Hepatitis B','Polio','Diphtheria '

third.plot.hexbin(x='Hepatitis B', y='Life expectancy ', gridsize=100);
third.plot.hexbin(x='Polio', y='Life expectancy ', gridsize=100);
third.plot.hexbin(x='Diphtheria ', y='Life expectancy ', gridsize=100);
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

X = d[['Life expectancy ']]
y=d[['Adult Mortality']]
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, X, y, cv=10)