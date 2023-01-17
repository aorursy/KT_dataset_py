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
col_names = ['Age', 'Op_Year', 'Axil_Nodes', 'Surv_Status']

data = pd.read_csv('/kaggle/input/habermans-survival-data-set/haberman.csv', names = col_names)

print(data.head(10))
data.shape # Dimensions of the Data Set
data.info()
data.describe() # Summary Stats of the Data Set
data.plot(subplots = True)
data.plot(kind = 'box')
sns.set()

data.plot(y = 'Age', kind = 'hist', bins = 22)

data.plot(y = 'Op_Year', kind = 'hist', bins = 12)

data.plot(y = 'Axil_Nodes', kind = 'hist')
sns.swarmplot(x = 'Surv_Status', y = 'Age', data = data)
sns.swarmplot(x = 'Surv_Status', y = 'Op_Year', data = data)
sns.swarmplot(x = 'Surv_Status', y = 'Axil_Nodes', data = data)
sns.violinplot(x = 'Surv_Status', y = 'Age', data = data)
sns.violinplot(x = 'Surv_Status', y = 'Op_Year', data = data)
sns.violinplot(x = 'Surv_Status', y = 'Axil_Nodes', data = data)
def ecdf(data):

    x = np.sort(data)

    y = np.arange(1, len(x)+1) / len(x)

    return (x,y)
x,y = ecdf(data['Age'])

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Age')

plt.ylabel('ECDF')

plt.margins(0.05) # Keeps data off plot edges

plt.show()
x,y = ecdf(data['Op_Year'])

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Year of Operation')

plt.ylabel('ECDF')

plt.margins(0.05) # Keeps data off plot edges

plt.show()
x,y = ecdf(data['Axil_Nodes'])

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Number of positive axillary nodes detected')

plt.ylabel('ECDF')

plt.margins(0.05) # Keeps data off plot edges

plt.show()
age_mean = np.mean(data['Age'])

age_std = np.std(data['Age'])

age_normal = np.random.normal(age_mean, age_std, size = 1000)

x,y = ecdf(age_normal)

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Age')

plt.ylabel('CDF')

plt.margins(0.05) # Keeps data off plot edges

plt.show()
year_mean = np.mean(data['Op_Year'])

year_std = np.std(data['Op_Year'])

year_normal = np.random.normal(year_mean, year_std, size = 1000)

x,y = ecdf(year_normal)

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Year of Operation')

plt.ylabel('CDF')

plt.margins(0.05) # Keeps data off plot edges

plt.show()
axil_mean = np.mean(data['Axil_Nodes'])

axil_std = np.std(data['Axil_Nodes'])

axil_normal = np.random.normal(axil_mean, axil_std, size = 1000)

x,y = ecdf(axil_normal)

plt.plot(x, y, marker='.', linestyle='none')

plt.xlabel('Number of positive axillary nodes detected')

plt.ylabel('CDF')

plt.margins(0.05) # Keeps data off plot edges

plt.show()
sns.pairplot(data)
sns.heatmap(data.corr())