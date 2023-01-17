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

pd.pandas.set_option('display.max_columns',None)

train = pd.read_csv('../input/covid19-symptoms-checker/Cleaned-Data.csv')

train.head()
countries = pd.get_dummies(train['Country'],drop_first=True)
train = pd.concat([train,countries],axis=1)

train.head()
train = train.drop(['Country'],axis=1)

train.head()
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

%matplotlib inline



data = train.iloc[0:30000,0:5]
data.head()
from sklearn.preprocessing import normalize

# data = train.copy()

data_scaled = normalize(data)

data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

data_scaled.head()
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))  

plt.title("Dendrograms")  

dend = shc.dendrogram(shc.linkage(data_scaled, method="complete", metric='euclidean'))


from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  

cluster.fit_predict(data_scaled)


plt.figure(figsize=(10, 7))  

plt.scatter(data_scaled['Sore-Throat'], data_scaled['Difficulty-in-Breathing'], c=cluster.labels_) 