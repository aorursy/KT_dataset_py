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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set_style('whitegrid')
df=pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')
df.head()
df.describe().transpose()
df.info()
sns.countplot(df['Legendary'])
col=['Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed']
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
sns.distplot(df['Total'])
sns.distplot(df['HP'])
sns.lmplot('Attack','Total',data=df,markers='.')
sns.jointplot(df['Attack'],df['Defense'],kind='hex')
sns.lmplot('Speed','Total',data=df,markers='.')
plt.figure(figsize=(12,6))
sns.countplot(df['Type 1'])
plt.figure(figsize=(12,6))
sns.countplot(df['Type 2'])
