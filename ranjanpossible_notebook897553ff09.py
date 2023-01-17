# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/HR.CSV"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np

data=pd.read_csv("../input/HR_comma_sep.csv")



data.info()

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
data['sales'].unique()
data['promotion_last_5years'].unique()
data['salary'].unique()
data.mean()
data.describe()
data.mean()['average_montly_hours']/30

print('# of people left = {}'.format(data[data['left']==1].size))

print('# of people stayed = {}'.format(data[data['left']==0].size))

print('protion of people who left in 5 years = {}%'.format(int(data[data['left']==1].size/data.size*100)))
corrmat = data.corr()

f, ax = plt.subplots(figsize=(4, 4))

# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
corrmat = data.corr()

f, ax = plt.subplots(figsize=(4, 4))

# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=.8, square=True)

plt.show()
corrmat_low = data[data['salary'] == 'low'].corr()

corrmat_medium = data[data['salary'] == 'medium'].corr()

corrmat_high = data[data['salary'] == 'high'].corr()



sns.heatmap(corrmat_low, vmax=.8, square=True,annot=True,fmt='.2f')
sns.heatmap(corrmat_medium, vmax=.8, square=True,annot=True,fmt='.2f')
sns.heatmap(corrmat_high, vmax=.8, square=True,annot=True,fmt='.2f')
data_low = data[data['salary'] == 'low']

data_medium = data[data['salary'] == 'medium']

data_high = data[data['salary'] == 'high']



print('# of low salary employees= ',data_low.shape[0])

print('# of medium salary employees= ',data_medium.shape[0])

print('# of high salary employees= ',data_high.shape[0])