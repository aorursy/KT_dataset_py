# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(0)
df = pd.read_csv('../input/countries of the world.csv', decimal = ',')
df.head()
df.info()
df.describe()
plt.rcParams["figure.figsize"] = [16, 12]
sns.heatmap(df.iloc[:,2:].dropna().corr(), annot=True, cmap="YlGnBu")
sns.jointplot(x = 'GDP ($ per capita)', y ='Net migration', data = df)
sns.jointplot(x = 'GDP ($ per capita)', y ='Literacy (%)', data = df, kind='hex', gridsize=20)
sns.boxplot(
    x='Region',
    y='Population',
    data=df
)
sns.kdeplot(df.Population)
sns.lmplot(x = 'GDP ($ per capita)', y ='Literacy (%)', data=df)
#sns.clustermap(df[:,[3,4]], metric="correlation")
df_subset = df.loc[:,['GDP ($ per capita)','Literacy (%)','Net migration']].dropna()
sns.clustermap(df_subset, metric="correlation", cmap="mako", robust=True)
