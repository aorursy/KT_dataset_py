# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Image('/kaggle/input/structureimage/__results___3_0.jpg', width = 400, height = 400)
SDD = pd.read_csv("/kaggle/input/cee-498-project9-structural-damage-detection/train.csv")
SDD.dtypes
SDD.shape
SDD.head()

SDD.columns
SDD.nunique(axis=0)
SDD.describe()
plt.plot(SDD["Condition"])
# calculate correlation matrix
corr = SDD.corr()# plot the heatmap
sns.heatmap( corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
num_cols = ['DA04','DA05','DA06','DA07','DA08','DA09', 'DA10', 'DA11', 'DA12', 'DA13']
plt.figure(figsize=(18,9))
SDD[num_cols].boxplot()
plt.title("Structural Damage Detection", fontsize=20)
plt.show()
plt.figure(figsize=(18,8))
plt.xlabel("DA04", fontsize=18)
plt.ylabel("DA07", fontsize=18)
plt.suptitle("Joint distribution of DA04 vs DA07", fontsize= 20)
plt.plot(SDD.DA04, SDD['DA07'], 'bo', alpha=0.2)
plt.show()
corr = data[['DA04','DA07']].corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
data.plot.scatter(x = 'DA04', y = 'DA07',
                  color = 'sandybrown', title = 'DA04 vs DA07')
damaged = SDD.loc[SDD['Condition'] == 1]
undamaged = SDD.loc[SDD['Condition'] == 0]
damaged.boxplot(figsize=(20,20))

undamaged.boxplot(figsize=(20,20))

undamaged.hist(figsize=(20,20))

damaged.hist(figsize=(20,20))

SDD, SDD1 = 'DA04', 'DA07'
data = pd.concat([damaged[SDD1], damaged[SDD]], axis=1)
data.plot.line(x = 'DA04', y = 'DA07', 
                             color = 'salmon', title = 'DA04 v DA07', ax = ax)

SDD, SDD1 = 'DA04', 'DA07'
data = pd.concat([undamaged[SDD1], undamaged[SDD]], axis=1)
data.plot.line(x = 'DA04', y = 'DA07', 
                             color = 'salmon', title = 'DA04 v DA07', ax = ax)
sns.pairplot()