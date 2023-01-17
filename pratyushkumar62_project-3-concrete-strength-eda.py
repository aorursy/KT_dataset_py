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

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import itertools

import warnings

warnings.filterwarnings('ignore')



data = pd.read_csv('/kaggle/input/cee-498-project3-forecast-strength-of-concrete/train.csv')
print(f"The given dataset contains {data.shape[0]} rows and {data.shape[1]} columns")

print(f"The given dataset contains {data.isna().sum().sum()} Null value")
data.head()
data.rename(columns=dict(zip(data.columns, ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',

       'fineagg', 'age', 'strength'])), inplace=True)
data.head()
data.describe().transpose()
plt.subplots(figsize=(12, 8))

ax = sns.boxplot(data=data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
import itertools



cols = [i for i in data.columns if i != 'strength']



fig = plt.figure(figsize=(15, 20))



for i,j in itertools.zip_longest(cols, range(len(cols))):

    plt.subplot(4,2,j+1)

    ax = sns.distplot(data[i],color='orange',rug=True)

    plt.axvline(data[i].mean(),linestyle="dashed",label="mean", color='black')

    plt.legend()

    plt.title(i)

    plt.xlabel("")
fig = plt.figure(figsize=(12, 8))

plt.axvline(data.strength.mean(),linestyle="dashed",label="mean", color='red')

sns.distplot(data.strength);
from scipy.stats import skew

numerical_features = data.select_dtypes(include=[np.number]).columns

categorical_features = data.select_dtypes(include=[np.object]).columns

skew_values = skew(data[numerical_features], nan_policy = 'omit')

dummy = pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 

           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)

dummy.sort_values(by = 'Skewness degree' , ascending = False)
g = sns.PairGrid(data)

g.map_upper(plt.scatter)

g.map_lower(sns.lineplot)

g.map_diag(sns.kdeplot, lw=3, legend=True);
plt.subplots(figsize=(12, 6))

corr = data.corr('spearman')



mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True



ax = sns.heatmap(data=corr, cmap='mako', annot=True, mask=mask)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45);
from mpl_toolkits.mplot3d import Axes3D



fig = plt.figure(figsize=(14,11))



ax  = fig.gca(projection = "3d")

#plt.subplot(111,projection = "3d") 



plot =  ax.scatter(data["cement"],

           data["strength"],

           data["superplastic"],

           linewidth=1,edgecolor ="k",

           c=data["age"],s=100,cmap="cool")



ax.set_xlabel("cement")

ax.set_ylabel("strength")

ax.set_zlabel("superplastic")



lab = fig.colorbar(plot,shrink=.5,aspect=5)

lab.set_label("AGE",fontsize = 15)



plt.title("3D plot for cement,compressive strength and super plasticizer",color="blue")

plt.show()