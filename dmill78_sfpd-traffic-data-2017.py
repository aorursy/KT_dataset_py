# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# %load ../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns
from sklearn.preprocessing import scale
import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
%matplotlib inline
plt.style.use('seaborn-white')
df = pd.read_csv('../input/sfpd2017.csv')
df.head()
# I renamed the file so that it was easier to load
df.info()

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# to find missing data in the data set
fig = plt.figure(figsize=(15,9))
fig.suptitle('SFPD demographic chart', fontsize=20)
sns.set_style('whitegrid')
sns.countplot(x='Race_description',hue='Sex',data=df,palette='rainbow')
sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=50)
sns.countplot(x='Race_description',data=df)

sns.countplot(x='Sex',data=df)

plt.figure(figsize=(12, 7))
sns.boxplot(x='Race_description',y='Age',data=df,palette='winter')
df = pd.DataFrame(np.random.randn(1000, 2), columns=['Race_description', 'Age'])
df.plot.hexbin(x='Race_description',y='Age',gridsize=25,cmap='Oranges')