# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (1).csv')
df2 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (2).csv')
df3 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile (3).csv')
df4 = pd.read_csv('../input/agricuture-crops-production-in-india/datafile.csv')
df5 = pd.read_csv('../input/agricuture-crops-production-in-india/produce.csv')
df3.head()
df3.describe(include='all')
df3.isnull().any()
print(df3.isnull().sum())
df3.info()
df4.isnull().any()
print(df4.isnull().sum())
df5.isnull().any()
print(df5.isnull().sum())
import seaborn as sns

sns.pairplot(data=df,kind="reg")
sns.distplot(df['Cost of Cultivation (`/Hectare) A2+FL'])
sns.heatmap(df.corr())
sns.jointplot(x='Yield (Quintal/ Hectare) ', y='Cost of Cultivation (`/Hectare) C2', data=df[df['Yield (Quintal/ Hectare) '] > 10], kind="reg", space=0, color="g")
df2.describe().T
Data2=df2.describe().T
dat=Data2.loc[:,"mean"]

dat.plot(kind='bar')
df2.head()
sns.pairplot(data=df2,kind="reg",y_vars=('Production 2006-07','Production 2007-08','Production 2008-09','Production 2009-10','Production 2010-11'),x_vars=('Area 2006-07','Area 2007-08','Area 2008-09','Area 2009-10','Area 2010-11'))