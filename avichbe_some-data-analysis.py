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
import seaborn as sns
#import matplotlib as plt
from matplotlib import pyplot as plt
import sklearn as skl
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
columns = df.columns
print(columns)
classes = df['diagnosis']
features = df.drop(['Unnamed: 32', 'diagnosis', 'id'], axis=1)

sns.countplot(classes, label='count')
features.describe()
data = pd.concat([classes, features.iloc[:, :6]], axis=1)
data = pd.melt(data, id_vars='diagnosis', var_name="features", value_name='value')
data
sns.swarmplot(x='features', y='value', hue='diagnosis', data=data)
features.corr
plt.figure(figsize=(20,20))
sns.heatmap(features.corr(), cmap="YlGnBu", annot=True)
sns.jointplot(features.loc[:,'concavity_worst'],
features.loc[:,'concave points_worst'], kind="reg")
plt.show()
data = pd.concat([classes, features.iloc[:,1:8]], axis=1)
sns.pairplot(data, hue='diagnosis', size=2.5)