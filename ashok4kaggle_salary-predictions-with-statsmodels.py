# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data Visulization library

import seaborn as sns #Data Visulizationn library

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Loading the Dataset

data = pd.read_csv('/kaggle/input/salary-data/DataSalary.csv')

data.head()
# Checking datasets

data.describe(include='all')
fig,ax = plt.subplots(figsize=(10,3))

sns.barplot(data['YearsExperience'],data['Salary'])
x1 = data['YearsExperience']  #Independent Variables

y = data['Salary'] #Dependent Variables
sns.scatterplot(x1,y)
# Regression Model

import statsmodels.api as sm

x = sm.add_constant(x1)

results = sm.OLS(y,x).fit()

results.summary()
yhat = 25790+x1*9449.9623

plt.scatter(x1,y)

fig = plt.plot(x1,yhat,c='red',label='linear regression')

plt.xlabel('Year of Experience',fontsize=15)

plt.ylabel('Salary',fontsize=15)
new_data = pd.DataFrame({'const':1,

                        'Year of Experience':[1.2,5.6,12.3,7.2]})

#new_data

new_data['Predicted Salary'] = results.predict(new_data)
new_data.drop('const',axis=1)