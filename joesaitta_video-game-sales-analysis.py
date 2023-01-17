# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(r'../input/Video_Games_Sales_as_at_30_Nov_2016.csv')
df.head()
df.columns
dg = df.groupby(['Year_of_Release', 'Platform'], as_index = False)['Global_Sales'].sum()
dp = dg.pivot('Year_of_Release','Platform','Global_Sales')

dp.plot(figsize = (9, 5))
dp.index
dp.head()
help(sns.lmplot)
sns.lmplot(x = 'Global_Sales', y = 'Critic_Score', data = df)
df[(df['Global_Sales'] > 80)]
sns.lmplot(x = 'Global_Sales', y = 'Critic_Score', data = df[(df['Global_Sales'] < 80)])
from scipy import stats
q = stats.pearsonr(x = df['Global_Sales'], y = df['Critic_Score'])