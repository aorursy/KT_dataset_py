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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 



%matplotlib inline

sns.set(color_codes=True)
estonia=pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
estonia.head()
estonia.describe()
estonia.info()
estonia['Age'].max()
estonia['Survived'].max()
estonia['Country'].max()
estonia[['Age','Survived']].corr()
y=estonia['Age']

x=estonia['Survived']

plt.scatter(x,y)
estonia['Age'].mean()
sns.distplot(estonia["Age"])
sns.jointplot(estonia['Age'],estonia['Survived'],kind='hex')

sns.pairplot(estonia[['Age','Survived']])
sns.stripplot(estonia['Age']>50,estonia['Survived'],jitter=True)
sns.distplot(estonia['Age']>50)
sns.lmplot(x='Age',y='Survived',data=estonia)