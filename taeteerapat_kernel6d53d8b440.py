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
# data analysis and wrangling
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import scipy.cluster.hierarchy as sch


# data analysis and wrangling
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from scipy import stats
color = sns.color_palette()

%matplotlib inline

pd.options.mode.chained_assignment = None

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



#การนำเข้าข้อมูล โดย import และประกาสค่าตัวแปรต่างๆ
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




# filename='/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv'
# T=pd.read_csv(filename,encoding='ISO-8859-1')
# T.head()
# filename='/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv'
# spoti=pd.read_csv(filename,encoding='ISO-8859-1')
# spoti.head(600)

# filename='/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv'
# T1=pd.read_csv(filename,encoding='ISO-8859-1')

T = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='latin1')
T1 = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='latin1')
T.head()
T.shape
T.isnull().sum()
T.isnull()
T=T.drop_duplicates()
T.shape

select_d = ['top genre','year','bpm','nrgy','dnce','val','dur','acous','spch','pop']
T = T[select_d]

select_d = ['top genre','year','bpm','nrgy','dnce','val','dur','acous','spch','pop']
T1 = T1[select_d]
T.head(600)
T.describe()
T['top genre'].value_counts().head()
sns.heatmap(T.corr(), annot=True)
print( len(T['top genre'].unique()) , "categories")

print("\n", T['top genre'].unique())
from sklearn.cluster import KMeans

sse = {}
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(T.drop(columns=['top genre']))
    T["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
kmeans = KMeans(n_clusters=10, max_iter=2000).fit(T.drop(columns=['top genre']))
T["clusters"] = kmeans.labels_
T.groupby(by=['clusters']).mean()
T.head(10)
sns.pairplot(T, vars=['year','bpm','nrgy','val','spch','pop'],hue='clusters',plot_kws={'alpha': .4});
sns.catplot(x="top genre", y="year", data=T)
sns.catplot(x="clusters", y="val", data=T)
sns.catplot(x="clusters", y="bpm", data=T)
sns.catplot(x="clusters", y="nrgy", data=T)


sns.catplot(x="clusters", y="dnce", data=T)


sns.catplot(x="clusters", y="dur", data=T)
sns.catplot(x="clusters", y="acous", data=T)
sns.catplot(x="clusters", y="spch", data=T)
sns.catplot(x="clusters", y="pop", data=T)
cols =['bpm','nrgy','dnce','val','acous','spch','pop','dur']
pt = preprocessing.PowerTransformer(method='yeo-johnson',standardize=True)
mat = pt.fit_transform(T1[cols])
mat[:5].round(4)
X=pd.DataFrame(mat, columns=cols)
X.head()
fig, ax=plt.subplots(figsize=(100,20))
dg=sch.dendrogram(sch.linkage(X,method='ward'),ax=ax,labels=T1['top genre'].values)
sns.clustermap(X,col_cluster=False,cmap="Blues")
hc=AgglomerativeClustering(n_clusters=8,linkage='ward')
hc
hc.fit(X)
hc.labels_
T1['cluster']=hc.labels_
T1.head()
T1.head(10)
T1.groupby('cluster').agg(['mean']).T
T1.groupby('cluster').head(2).sort_values('cluster')
cols =['bpm','nrgy','dnce','val','acous','spch','pop','dur']
fig,ax = plt.subplots(nrows=4,ncols=2,figsize=(20,9))
ax=ax.ravel()
for i, col in enumerate(cols):
    sns.violinplot(x='cluster',y=col,data=T1,ax=ax[i])