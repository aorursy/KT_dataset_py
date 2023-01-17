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
import matplotlib.pyplot as plt

import seaborn as sns
elephants = pd.read_csv('../input/elephantsmf/ElephantsMF.csv', index_col=0)

elephants.head()
elephants[['Age', 'Height']].describe()
elephants['Sex'].value_counts()
elephants['Sex'].value_counts(ascending=True).plot(kind='pie', figsize=(16, 8))
fig, ax =plt.subplots(1,2, figsize=(16, 8))



sns.boxplot(x=elephants['Sex'], y=elephants['Age'], ax=ax[0])

sns.boxplot(x=elephants['Sex'], y=elephants['Height'], ax=ax[1])



fig.show()
fig, ax =plt.subplots(1,2, figsize=(16, 8))



sns.distplot(a=elephants['Age'], kde=False, ax=ax[0])

sns.distplot(a=elephants['Height'], kde=False, ax=ax[1])



fig.show()
elephants['Age'][elephants['Sex']== 'M']
sns.distplot(a=elephants['Age'][elephants['Sex']== 'M'], kde=False, label='Male')

sns.distplot(a=elephants['Age'][elephants['Sex']== 'F'], kde=False, label='Female')



plt.title('Distribution of Elephants Age')

plt.legend()
grp_sex = elephants.groupby(['Sex'], as_index=False).mean()

grp_sex
elephants[['Age', 'Height']].corr()
sns.scatterplot(x=elephants['Age'], y=elephants['Height'])
sns.scatterplot(x=elephants['Age'], y=elephants['Height'], hue=elephants['Sex'])