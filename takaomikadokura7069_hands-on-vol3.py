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

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#データの読み込み

train = pd.read_csv("/kaggle/input/uci-wholesale-customers-data/Wholesale customers data.csv")
print(train.shape)
train.head()
train.info()
train.describe()
# import pandas_profiling as pdp

# pdp.ProfileReport(train)
# 不要なカラムを削除(販売チャネル、各顧客の地域)

del(train['Channel'])

del(train['Region'])

train.head()
from sklearn.cluster import KMeans



# k-meansを使用し、4分割する

pred = KMeans(n_clusters=4).fit_predict(train)



# クラスターIDを元データに設定

train['cluster_id'] = pred

#クラスターIDごとの件数を表示

train.groupby('cluster_id')['cluster_id'].count()
g=sns.FacetGrid(train,col="cluster_id")

g=g.map(sns.distplot,"Fresh")

g.add_legend()
#クラスターIDごとに集計する

train_sum = train.groupby('cluster_id').sum()

train_sum.head()
#積み上げ棒グラフをクラスターID別に表示

train_sum.plot.bar(stacked=True)
train_sum_100p = train_sum.apply(lambda x:x/sum(x),axis=1)

train_sum_100p
#積み上げ棒グラフをクラスターID別に表示

train_sum_100p.plot.bar(stacked=True)
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster



#ユークリッド距離とウォード法を使用して階層型クラスタリングを行う

Z = linkage(train, method='ward', metric='euclidean')

pd.DataFrame(Z)

import matplotlib.pyplot as plt

fig2, ax2 = plt.subplots(figsize=(20,5))



# 樹形図を作成

ax2 = dendrogram(Z)

fig2.show()

from scipy.cluster.hierarchy import fcluster



# クラスタ数を指定してクラスタリング

clusters = fcluster(Z, t=3, criterion='maxclust')

for i, c in enumerate(clusters):

    print(i, c)

# エルボー方による推定。クラスター数を1から10に増やして、それぞれの距離の総和を求める

dist_list =[]

for i in range(1,10):

    kmeans= KMeans(n_clusters=i, init='random', random_state=0)

    kmeans.fit(train)

    dist_list.append(kmeans.inertia_)

    

# グラフを表示

plt.plot(range(1,10), dist_list,marker='+')

plt.xlabel('Number of clusters')

plt.ylabel('Distortion')
