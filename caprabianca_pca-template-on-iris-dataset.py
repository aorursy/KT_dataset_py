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
df = pd.read_csv("../input/Iris.csv", index_col=0)

df.head()
X = df.iloc[:,0:4].values

y = df.iloc[:,4].values
from sklearn import decomposition



pca = decomposition.PCA(n_components=4)



pc = pca.fit_transform(X)
pc_df = pd.DataFrame(data = pc , 

        columns = ['PC1', 'PC2', 'PC3', 'PC4'])

pc_df['Cluster'] = y

pc_df.head()
pca.explained_variance_ratio_
import seaborn as sns



#Scree plot

df = pd.DataFrame({'var':pca.explained_variance_ratio_,

             'PC':['PC1','PC2','PC3','PC4']})

sns.barplot(x='PC',y="var", 

           data=df, color="c");
sns.lmplot( x="PC1", y="PC2",

  data=pc_df, 

  fit_reg=False, 

  hue='Cluster', # color by cluster

  legend=True,

  scatter_kws={"s": 80}) # specify the point size