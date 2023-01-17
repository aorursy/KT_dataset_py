# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# setting the font size

import matplotlib

matplotlib.rcParams.update({'font.size': 12})



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/iris/Iris.csv', index_col='Id')

print(data.shape)

data.head()
data.info()
data.describe()
target = 'Species'

numerical_variables = [column for column in data.columns if column != target]

print(f'Numerical variables: \n{numerical_variables}')

print(f'Target: {target}')

data[numerical_variables].plot(kind='hist', 

                               bins=30,

                              subplots=True,

                              figsize=(10,20),

                              grid=True,

                            sharex=False,

                              legend='Distribution of the variables')
t = data[target].value_counts().rename('').plot(kind='pie')
data[numerical_variables].plot(kind='box', 

          subplots=True,

          figsize=(10,10),

          grid=True,

          legend='Outliers detection',

         sharex=False,

         vert=False,

         layout=(4,1))
for variable in numerical_variables:

    data.plot(kind='scatter', 

              x=variable, 

              y=target)
for variable in numerical_variables:

    data.groupby(target)[variable].plot(kind='hist',

                                   bins=30,

                                  grid=True,                            

                                  legend='Distribution by specie')

    plt.xlabel(variable)

    plt.show()
sns.heatmap(data=data.corr(), annot=True)