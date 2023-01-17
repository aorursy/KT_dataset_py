import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)  

import cufflinks as cf  

cf.go_offline() 

import os

df = pd.read_csv('../input/google_review_ratings.csv')

df.head()
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.info()
Cols = [str(i) for i in range(1,25)]

Cols =['Category '+i for i in Cols]
for i in Cols:

    df[i] = pd.to_numeric(df[i],errors = 'coerce')
df.info()
df.isnull().sum()
df = df.fillna(df.mean())
df.isnull().sum()
df.describe()
New_cols = ['user_id', 'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 'zoo', 'restaurants', 'pubs_bars', 'local_services', 'burger_pizza_shops', 'hotels_other_lodgings', 'juice_bars', 'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas', 'cafes', 'view_points', 'monuments', 'gardens']

df.columns = New_cols
x = df.copy()

new = x['user_id'].str.split(' ',n=2,expand=True)

x['user'] = new[0]

x['id'] = new[1]

x = x.drop(['user_id','user'],axis=1)

x.head()
AvgR = df[New_cols[1:]].mean()

AvgR = AvgR.sort_values()

plt.figure(figsize=(10,7))

plt.barh(np.arange(len(New_cols[1:])), AvgR.values, align='center')

plt.yticks(np.arange(len(New_cols[1:])), AvgR.index)

plt.ylabel('Categories')

plt.xlabel('Average Rating')

plt.title('Average Rating for every Category')
New_cols.remove('user_id')
df[New_cols].iplot(kind='box')
vals = df.iloc[ :, 1:].values



from sklearn.cluster import KMeans

wcss = []

for ii in range( 1, 30 ):

    kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( vals )

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
X = df.drop(['user_id'],axis=1).values

Y = df['user_id'].values
km = KMeans(n_clusters=4, init="k-means++", n_init=10, max_iter=500) 

y_pred = kmeans.fit_predict(X)
df["Cluster"] = y_pred

cols = list(df.columns)

cols.remove("user_id")



sns.pairplot( df[cols], hue="Cluster")
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import scale as s

from scipy.cluster.hierarchy import dendrogram, linkage
Z = sch.linkage(x,method='ward')

den = sch.dendrogram(Z)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 

plt.title('Hierarchical Clustering')
def fd(*args, **kwargs):

    max_d = kwargs.pop('max_d', None)

    if max_d and 'color_threshold' not in kwargs:

        kwargs['color_threshold'] = max_d

    annotate_above = kwargs.pop('annotate_above', 0)



    ddata = dendrogram(*args, **kwargs)



    if not kwargs.get('no_plot', False):

        plt.title('Hierarchical Clustering Dendrogram (truncated)')

        plt.xlabel('sample index or (cluster size)')

        plt.ylabel('distance')

        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):

            x = 0.5 * sum(i[1:3])

            y = d[1]

            if y > annotate_above:

                plt.plot(x, y, 'o', c=c)

                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),

                             textcoords='offset points',

                             va='top', ha='center')

        if max_d:

            plt.axhline(y=max_d, c='k')

    return ddata
Z = linkage(x,method='ward')

fd(Z,leaf_rotation=90.,show_contracted=True,annotate_above=30000,max_d=80000)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 