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



url = 'https://raw.githubusercontent.com/krishnaik06/Multicollinearity/master/data/Advertising.csv'

df = pd.read_csv(url, index_col=0)

df.head(5)
x=df.iloc[:,:-1]

y=df.iloc[:,-1]
import seaborn as sns

feature_corr=df.corr()

index=feature_corr.index

sns.heatmap(df[index].corr())
import statsmodels.api as sm

x=sm.add_constant(x)

x
models=sm.OLS(y,x).fit()
models.summary()
import matplotlib.pyplot as plt

x.iloc[:,1:].corr()
df1=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/Multicollinearity/master/data/Salary_Data.csv')

df1.head()
x=df1.iloc[:,:-1]

y=df1.iloc[:,-1]


x=sm.add_constant(x)
modell=sm.OLS(y,x).fit()

modell.summary()
x.iloc[:,1:].corr()