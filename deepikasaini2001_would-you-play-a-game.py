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
import pandas as pd

data = pd.read_excel('/kaggle/input/sum-of-six-dice/Probabilities of sum of numbers when 6 die are thrown.xlsx')
data
# Let's sort the data and set 'SUM' as its index

data.sort_values(by='SUM',inplace=True)

data.set_index('SUM', inplace=True)
# Let's check

data.head()
import seaborn as sns
sns.scatterplot(x = data.index , y=data['Probabilities'])
sns.barplot(x = data.index , y=data['Probabilities'])
data['Probabilities'].sum()

# Very close to 1
more_likely = data[data['Probabilities'] >= 0.03]

less_likely = data[data['Probabilities'] < 0.03]
more_likely
less_likely
# Let's count the number of sums of both the groups

print('Number of Sums from more likely group is ', more_likely.count()[0])

print('Number of Sums from less likely group is ', less_likely.count()[0])

# Sum of the probabilities of both

print('Probability of getting Sums from more likely group is ', more_likely.sum()[0])

print('Probability of getting Sums from less likely group is ', less_likely.sum()[0])
data['Amount'] = 0

for i in data.index:

    if data.loc[i,'Probabilities'] >=0.03:

        data.loc[i,'Amount'] = -500

    else:

        data.loc[i,'Amount'] = 1000
# Expectation 

more_likely.sum()[0] * -500 + less_likely.sum()[0]*1000