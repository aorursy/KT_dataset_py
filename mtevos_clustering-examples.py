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
os.chdir('/kaggle/input/sberbank-russian-housing-market')

from zipfile import ZipFile



zip_file = ZipFile('train.csv.zip')

dfs = {text_file.filename: pd.read_csv(zip_file.open(text_file.filename)) for text_file in zip_file.infolist()

       if text_file.filename.endswith('.csv')}

df = dfs['train.csv']

df.shape

# os.listdir()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt





from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



from mpl_toolkits.mplot3d import Axes3D

from collections import Counter

df.head()
# df.shape
Counter(df.dtypes)
df.select_dtypes(include='O').head()
df.groupby('product_type')['sub_area'].count()
majors = [col.split('_')[0] for col in list(df)]

pd.DataFrame(Counter(majors), index = ['val']).T.sort_values('val', ascending = False).head(20)
df[[col for col in list(df) if 'church' in col]]
sns.scatterplot(df.church_synagogue_km, df.mosque_km);
df[[col for col in list(df) if 'cafe' in col]]
df.groupby('sub_area')['cafe_avg_price_500'].median()
df.price_doc.hist(bins = 300)

plt.xlim(0,4*1e7);
df['ppsm'] = df.price_doc / (df.full_sq + 1)
sns.distplot(df.ppsm.fillna(-1), bins = 300, kde = False)

plt.title('PRICE PER SQUARE METER')

plt.xlim(0,0.4*1e6);
print(f'median price of 1 sq meter in Moscow in 2011 {int(1.326463e+05/30)} USD')
pd.DataFrame(df.groupby('sub_area')['ppsm'].median()).reset_index().sort_values('ppsm', ascending = False).head(20)
pd.DataFrame(df.groupby('sub_area')['ppsm'].median()).reset_index().sort_values('ppsm', ascending = False).tail(20)
# df.full_sq.describe()
df.full_sq.hist(bins = 300)

plt.xlim(0,400);
df.life_sq.hist(bins = 300)

plt.xlim(0,400);
sns.scatterplot(data = df, x = 'full_sq', y = 'life_sq')

plt.xlim(0,500)

plt.ylim(0,500);
df[df.life_sq > df.full_sq].groupby('product_type')['ppsm'].median()
df.floor.hist(bins = 300)

plt.xlim(0,50);
ndf = df.fillna(0)

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(ndf[[col for col in list(df) if 'cafe' in col]])

df['tsne-2d-one'] = tsne_results[:,0]

df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
# df.sub_area.nunique()
from pylab import rcParams

rcParams['figure.figsize'] = 15, 15

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    hue="sub_area",

    palette=sns.color_palette("hls", df.sub_area.nunique()),

    data=df,

    legend=None,

    alpha=0.3);
from pylab import rcParams

rcParams['figure.figsize'] = 20, 20

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    hue="product_type",

    palette=sns.color_palette("hls", df.product_type.nunique()),

    data=df,

#     legend='Full',

    alpha=0.3);
sdf = df.sample(n = 1000)

fig, ax = plt.subplots()

ax.scatter(sdf["tsne-2d-one"], sdf["tsne-2d-two"])



for i, txt in sdf.iterrows():

    ax.annotate(txt[['sub_area']].values[0], (txt[["tsne-2d-one"]].values[0], txt[["tsne-2d-two"]].values[0]))
featcols = list(df.select_dtypes(exclude='O'))[1:-2]
len(featcols)
pca = PCA(n_components=3)

pca_result = pca.fit_transform(ndf[featcols].values)

df['pca-one'] = pca_result[:,0]

df['pca-two'] = pca_result[:,1] 

df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
indices = ['PC-1','PC-2', 'PC-3']

pcs = pd.DataFrame(pca.components_,columns=featcols,index = ['PC-1','PC-2', 'PC-3']).T

for p in indices:

    pcs[p] = pcs[p].apply(lambda x: abs(x))

# pcs[indices] = pcs[indices] * 1e25

pcs.sort_values(indices[0], ascending = False).head(20)
# APPLYING NORMALIZATION AND REDOING STUFF

from sklearn.preprocessing import normalize

nndf = pd.DataFrame(normalize(df[featcols].fillna(-1)), columns= featcols)
#DROPPING PRICE
pca = PCA(n_components=3)

pca_result = pca.fit_transform(nndf[featcols[:-2]].values)

df['pca-one'] = pca_result[:,0]

df['pca-two'] = pca_result[:,1] 

df['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
indices = ['PC-1','PC-2', 'PC-3']

pcs = pd.DataFrame(pca.components_,columns=featcols[:-2],index = ['PC-1','PC-2', 'PC-3']).T

for p in indices:

    pcs[p] = pcs[p].apply(lambda x: abs(x))

# pcs[indices] = pcs[indices] * 1e25

pcs.sort_values(indices[0], ascending = False).head(20)
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()
idf = df[df.product_type != 'Investment']

ridf = pd.DataFrame(rb.fit_transform(idf[featcols[:-2]].fillna(-1)), columns= featcols[:-2])

ridf['price'] = idf['price_doc'].copy(deep = True)
idf.full_sq.describe(), ridf.full_sq.describe()
from sklearn.mixture import GaussianMixture as GM

from sklearn.cluster import DBSCAN

from sklearn.metrics import davies_bouldin_score, silhouette_score
vals = []

ft= featcols[:-2]

for i in range(2,30):

    gm = GM(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(ridf[ft])

    idx = {'IDX': i,

           'BIC': gm.bic(ridf[ft]),

           'BDS': davies_bouldin_score(ridf[ft], gm.predict(ridf[ft])),

            'SS': silhouette_score(ridf[ft], gm.predict(ridf[ft]))}

    vals.append(idx)
scores = pd.DataFrame(vals)
plt.plot(scores['SS'], color = 'red')

plt.plot(scores['BDS'], color = 'orange')
plt.plot(scores['BIC'], color = 'blue')
# vals = []

ft= ['life_sq', 'floor',

 'max_floor',

 'num_room',]
nn = df[ft+['product_type']].dropna()
pca = PCA(n_components=3)

pca_result = pca.fit_transform(normalize(nn[ft].values))

nn['pca-one'] = pca_result[:,0]

nn['pca-two'] = pca_result[:,1] 

nn['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
pcas = ['pca-one', 'pca-two', 'pca-three']
indices = ['PC-1','PC-2', 'PC-3']

pcs = pd.DataFrame(pca.components_,columns=ft,index = ['PC-1','PC-2', 'PC-3']).T

for p in indices:

    pcs[p] = pcs[p].apply(lambda x: abs(x))

# pcs[indices] = pcs[indices] * 1e25

pcs.sort_values(indices[0], ascending = False).head(20)
vals = []

for i in range(2,8):

    gm = GM(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(nn[pcas])

    idx = {'IDX': i,

           'GMM':gm,

           'BIC': gm.bic(nn[pcas]),

           'BDS': davies_bouldin_score(nn[pcas], gm.predict(nn[pcas])),

            'SS': silhouette_score(nn[pcas], gm.predict(nn[pcas]))}

    vals.append(idx)

scores = pd.DataFrame(vals)
plt.plot(scores['SS'], color = 'red')

plt.plot(scores['BDS'], color = 'orange');
dbsc = DBSCAN()

dbsc.__dict__
for ep in [0.1, 0.2, 0.3]:

    dbsc = DBSCAN(eps = ep)

    dbsc.fit(normalize(nn[ft]))

    nn[f'dbc{ep}'] = dbsc.labels_
nn['dbc0.3'].unique()
for col in list(nn):

    if 'dbc' in col:

        print(f' {col} number of clusters {nn[col].nunique()} \n unclusterables {nn[nn[col] == -1].shape[0]} \n silhouette score: { silhouette_score(nn[ft], nn[col])}\n\n')
sns.scatterplot(

    x="pca-one", y="pca-two",

    hue="dbc0.1",

    palette=sns.color_palette("hls", 7),

    data=nn,

    legend='full',

    alpha=0.3

)



plt.show()
nn.groupby('dbc0.1')[ft].agg(['median', 'mean', 'min', 'max']).T