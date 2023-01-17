import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/Seed_Data.csv')
df.head()
print('Numbers of rows {} and number of columns {} '.format(df.shape[0], df.shape[1]))
print('\n')
df.info()
df.describe()
import warnings
warnings.filterwarnings("ignore")

sns.set(style="darkgrid")
sns.lmplot('A','C',data=df, hue='target',
           palette='Set1',size=7,aspect=1.2,fit_reg=False);
sns.lmplot('A','A_Coef',data=df, hue='target',
           palette='Set1',size=7,aspect=1.2,fit_reg=False);
g = sns.FacetGrid(data = df, hue='target', palette='Set2', size=7, aspect=3)
g = g.map(plt.hist,'A',bins=22,alpha=0.6)
plt.legend();
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(df.drop('target',axis=1))
centers = kmeans.cluster_centers_
centers
df['klabels'] = kmeans.labels_
df.head()
f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True,figsize = (12,8) )

# For fitted with kmeans 
ax1.set_title('K Means (K = 3)')
ax1.scatter(x = df['A'], y = df['A_Coef'], 
            c = df['klabels'], cmap='rainbow')
ax1.scatter(x=centers[:, 0], y=centers[:, 5],
            c='black',s=300, alpha=0.5);

# For original data 
ax2.set_title("Original")
ax2.scatter(x = df['A'], y = df['A_Coef'], 
            c = df['target'], cmap='rainbow')
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
X = df.iloc[:, [0,1,2,3,4,5,6]].values

scaler = StandardScaler()
X = scaler.fit_transform(X)
hc = AgglomerativeClustering(n_clusters= 3, affinity= 'euclidean', linkage= 'ward')
previsoes = hc.fit_predict(X)
fig = plt.figure(figsize=(12,9))
fig = dendograma = dendrogram(linkage(previsoes, method= 'ward'), color_threshold=1, show_leaf_counts=True,
                             truncate_mode='lastp')
df.klabels.value_counts()
df.target.value_counts()
sum_square = {}

# Let's test for K from 1 to 10
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k).fit(df.drop('target',axis=1))
    
    sum_square[k] = kmeans.inertia_ 
plt.plot(list(sum_square.keys()), list(sum_square.values()),
         linestyle ='-', marker = 'H', color = 'g',
         markersize = 8,markerfacecolor = 'b');
