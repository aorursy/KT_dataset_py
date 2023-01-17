import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as pl
DATA_DIR = '../input'

DATA_FILE = 'dataset.csv'
# data_df = pd.read_csv(os.path.join(DATA_DIR, DATA_FILE), na_values = {'?'})

data_df = pd.read_csv('../input/dmassign1/data.csv', na_values = {'?'})

# data_df = pd.read_csv('data.csv')
data_df.head()
data_df.describe()
# data_df.Col31.unique()[500:]
data_df.dtypes['Col187']
null_count = data_df.isnull().sum()

print(null_count[null_count != 0][:50])

print(null_count[null_count != 0][50:])
data_df.dtypes[150:]
# test_df = pd.DataFrame([[1, 'alpha'], [np.nan, None]])

# test_df.isna()
data_df['Col197'].replace(['sm', 'me', 'M.E.', 'la'], ['SM', 'ME', 'ME', 'LA'], inplace = True)
print(data_df['Col189'].unique())

print(data_df['Col190'].unique())

print(data_df['Col191'].unique())

print(data_df['Col192'].unique())

print(data_df['Col193'].unique())

print(data_df['Col194'].unique())

print(data_df['Col195'].unique())

print(data_df['Col196'].unique())

print(data_df['Col197'].unique())
data_df_dum = pd.get_dummies(data_df, columns = ['Col192', 'Col193', 'Col194'])

data_df_dum.loc[:, 'Col189':].columns
data_df_cat = data_df_dum.replace({'Col189':{'no':0, 'yes':1},

                                   'Col190':{'sacc1':1, 'sacc2':2, 'sacc4':4, 'sacc5':5},

                                   'Col191':{'time1':1, 'time2':2, 'time3':3},

#                                    'Col192':{'p1':1, 'p2':2, 'p3':3, 'p4':4, 'p5':5, 'p6':6, 'p7':7, 'p8':8, 'p9':9, 'p10':10},

                                   'Col195':{'Jb1':1, 'Jb2':2, 'Jb3':3, 'Jb4':4},

                                   'Col196':{'H1':1, 'H2':2, 'H3':3},

                                   'Col197':{'SM':1, 'ME':2, 'LA':3, 'XL':4}

                                  })
data_df_cat.dtypes.loc['Col188':]
data_df_cat.iloc[:1299, :].corr()['Class'][-25:]
data_df_cat.iloc[:1299, :].corr()['Class'].abs().sort_values(ascending = False)[:14]
# class_corr = dict(data_df.iloc[:1299, :].corr()['Class'])

features = data_df_cat.iloc[:1299, :].corr()['Class'].abs().sort_values(ascending = False)[:53].index
features
data_df['Class'].unique()
data_df_cat.columns[1:]
# from sklearn.preprocessing import StandardScaler



# scale1 = StandardScaler()

# data_df_cat_norm = pd.DataFrame(scale1.fit_transform(data_df_cat.iloc[:, 1:].values), columns = data_df_cat.columns[1:])
# data_df_cat_norm.iloc[:1299, :].corr()['Class'].abs().sort_values(ascending = False)[:53]
import seaborn as sns



sns.distplot(data_df.iloc[:1299, :].Class)
data_df_cat.mean()
null_cols = data_df_cat.columns[data_df_cat.isnull().any()]

null_cols
num_null_cols = null_cols[:-4]

cat_null_cols = null_cols[-4:-1]

print(num_null_cols, cat_null_cols)
data_df_cat[num_null_cols] = data_df_cat[num_null_cols].fillna(data_df_cat[num_null_cols].mean())
# df = pd.DataFrame([[1, 2, 3], [np.nan, np.nan, 5], [np.nan, 4, np.nan]])

# df.fillna(pd.Series([6, 7, 8]))
# pd.Series(np.squeeze(data_df_cat[cat_null_cols].mode().values))

data_df_cat[cat_null_cols].mode().loc[0, :]
data_df_cat[cat_null_cols] = data_df_cat[cat_null_cols].fillna(data_df_cat[cat_null_cols].mode().loc[0, :])
data_df_cat.columns[data_df_cat.isnull().any()]
# Drop nulls and normalise

data_features = data_df_cat[features]

data_class = data_features['Class']

data_features.drop(columns = 'Class', inplace = True)
from sklearn.preprocessing import StandardScaler



scale = StandardScaler()

data_features_norm = scale.fit_transform(data_features.values)
# data_features.info()
data_features.isnull().sum().sum()
data_features.dropna().info()

# dropping null rows seems like a reasonable solution, losing 94 entries.

# Can't drop any rows as we need a prediction for all the rows

data_feat = data_features.dropna()
# print(data_features.values.shape)

print(data_features_norm.shape)

# data_feat.columns

# data_features_norm[:5, :]
from sklearn.cluster import KMeans



km = KMeans(n_clusters = 5)

km.fit(data_features_norm)
pred = km.predict(data_feat.values[:1300, :])
pred[:100]
# data_df.loc[:100, 'Class'].values

data_class.iloc[:100].astype(int).values
from sklearn.cluster import MiniBatchKMeans



mbkm = MiniBatchKMeans(n_clusters=5)

mbkm.fit(data_features)
pred_mbkm = mbkm.predict(data_feat.values[:1300, :])

pred_mbkm[:100]
from sklearn.cluster import AgglomerativeClustering



ag = AgglomerativeClustering(n_clusters=5, linkage = 'complete')

pred_ag = ag.fit_predict(data_features_norm)
pred_ag[:100]
data_features_norm.max()
from sklearn.cluster import Birch



bi = Birch(n_clusters=5)

bi.fit(data_features_norm)
pred_bi = bi.predict(data_features[:1300])

pred_bi[:100]
data_class.iloc[:100].astype(int).values
data_df_cat.columns[-20:]
from sklearn.decomposition import PCA



pca = PCA(n_components=3)

x = data_df_cat.iloc[:1300, :].drop(columns = ['ID', 'Class'])

y = data_df_cat.loc[:1299, 'Class']

x_plot = pca.fit_transform(x.values)
x_plot.shape
# def colour_plot(y, x):

%matplotlib

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(1, 6):

    pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

pl.show()

    # pl.scatter(x_plot[:, 0], x_plot[:, 1])

    

# colour_plot(y, 0)



from mpl_toolkits.mplot3d import Axes3D

%matplotlib

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

fig = pl.figure()

ax = fig.add_subplot(111, projection='3d')



for i in np.arange(2, 4):

#     pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

    ax.scatter(x_plot[y == i, 0], x_plot[y == i, 1], x_plot[y == i, 2], c = colors[i])

ax.view_init(0, 135)

pl.show()
x_plot[pred_bi == 1, 0]
%matplotlib 

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(0, 5):

    pl.scatter(x_plot[pred_bi == i, 0], x_plot[pred_bi == i, 1], c = colors[i+1])

pl.show()



# colour_plot(pred_bi, 1)
colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(0, 5):

    pl.scatter(x_plot[pred == i, 0], x_plot[pred == i, 1], c = colors[i+1])

pl.show()



# colour_plot(pred)
data_df_cat.columns.drop('ID')
# data_df_cat.drop(columns = ['ID', 'Class']).isnull().sum().sum()

from sklearn.preprocessing import StandardScaler



scale1 = StandardScaler()

data_df_cat_norm = pd.DataFrame(scale1.fit_transform(data_df_cat.drop(columns = ['ID', 'Class']).values), columns = data_df_cat.columns.drop(['ID', 'Class']))
data_df_cat_norm.shape
pca_full = PCA(n_components = 20)

pca_features = pca_full.fit_transform(data_df_cat_norm.values)
pca_features.shape
bi_pca = Birch(n_clusters=5)

bi_pca.fit(pca_features)
pred_bi_pca = bi_pca.predict(pca_features[:1300, :])

pred_bi_pca[:100]
%matplotlib 

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(0, 5):

    pl.scatter(pca_features[:1300, :][pred_bi_pca == i, 0], pca_features[:1300, :][pred_bi_pca == i, 1], c = colors[i+1])

pl.show()
data_df_cat_norm.max()
ag_pca = AgglomerativeClustering(n_clusters=5, linkage = 'complete')

pred_ag_pca = ag_pca.fit_predict(pca_features)
pred_ag_pca.shape
x_plot.shape
%matplotlib 

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(0, 5):

    pl.scatter(pca_features[:1300, :][pred_ag_pca[:1300] == i, 0], pca_features[:1300, :][pred_ag_pca[:1300] == i, 1], c = colors[i+1])



pl.show()
%matplotlib



colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

fig = pl.figure()

ax = fig.add_subplot(111, projection='3d')

for i in np.arange(0, 5):

#     pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

    ax.scatter(x_plot[pred_ag_pca[:1300] == i, 0], x_plot[pred_ag_pca[:1300] == i, 1], x_plot[pred_ag_pca[:1300] == i, 2], c = colors[i+1])

ax.view_init(0, 135)

pl.show()
%matplotlib

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in np.arange(1, 6): #[5, 4, 3, 2, 1]:

    pl.scatter(pca_features[:1300, :][y == i, 0], pca_features[:1300, :][y == i, 1], c = colors[i])

pl.show()
colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

fig = pl.figure()

ax = fig.add_subplot(111, projection='3d')



for i in [5, 4, 3, 2, 1]:#np.arange(1, 6):

#     pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

    ax.scatter(pca_features[:1300, :][y == i, 0], pca_features[:1300, :][y == i, 1], pca_features[:1300, :][y == i, 2], c = colors[i])

ax.view_init(0, 0)

pl.show()
pca_pre = PCA(n_components=15, whiten=True)

pca_pre_features = pca_pre.fit_transform(data_df_cat_norm.values)

pca_pre_features.shape
pca_pre.explained_variance_ratio_.sum()
from sklearn.manifold import TSNE



tsne = TSNE(n_components=3)

tsne_features = tsne.fit_transform(pca_pre_features)

tsne_features.shape
tsne_features.shape
y.shape
tsne_features[:1300, :][y == i, 0].shape
%matplotlib

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in [1, 4, 5, 2, 3]: # np.arange(1, 6): 

    pl.scatter(tsne_features[:1300, :][y == i, 0], tsne_features[:1300, :][y == i, 1], c = colors[i])

pl.show()
colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

fig = pl.figure()

ax = fig.add_subplot(111, projection='3d')



for i in np.arange(1, 6):

#     pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

    ax.scatter(tsne_features[:1300, :][y == i, 0], tsne_features[:1300, :][y == i, 1], tsne_features[:1300, :][y == i, 2], c = colors[i])

ax.view_init(0, 0)

pl.show()
bi_tsne = Birch(n_clusters=5, threshold=0.3)

bi_tsne.fit(tsne_features)
pred_bi_tsne = bi_tsne.predict(tsne_features[:1300, :])

pred_bi_tsne[:100]
y[:100].astype(int).values
%matplotlib 

colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

for i in [0, 3, 2, 1, 4]: #np.arange(0, 5):

    pl.scatter(tsne_features[:1300, :][pred_bi_tsne == i, 0], tsne_features[:1300, :][pred_bi_tsne == i, 1], c = colors[i+1])

pl.show()
colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

fig = pl.figure()

ax = fig.add_subplot(111, projection='3d')



for i in np.arange(0, 5):

#     pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

    ax.scatter(tsne_features[:1300, :][pred_bi_tsne == i, 0], tsne_features[:1300, :][pred_bi_tsne == i, 1], tsne_features[:1300, :][pred_bi_tsne == i, 2], c = colors[i+1])

ax.view_init(0, 0)

pl.show()
bi_tsne.predict(tsne_features).shape
data_df.ID
# ag_tsne = AgglomerativeClustering(n_clusters=5, linkage = 'complete')

# pred_ag_tsne = ag_tsne.fit_predict(tsne_features)
# pred_ag_tsne.shape
# %matplotlib 

# colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

# for i in [1, 4, 0, 2, 3]: #np.arange(0, 5):

#     pl.scatter(tsne_features[:1300, :][pred_ag_tsne[:1300] == i, 0], tsne_features[:1300, :][pred_ag_tsne[:1300] == i, 1], c = colors[i+1])

# pl.show()
# colors = {1:'red', 2:'orange', 3:'yellow', 4:'green', 5:'blue'}

# fig = pl.figure()

# ax = fig.add_subplot(111, projection='3d')



# for i in np.arange(0, 5):

# #     pl.scatter(x_plot[y == i, 0], x_plot[y == i, 1], c = colors[i])

#     ax.scatter(tsne_features[:1300, :][pred_ag_tsne[:1300] == i, 0], tsne_features[:1300, :][pred_ag_tsne[:1300] == i, 1], tsne_features[:1300, :][pred_ag_tsne[:1300] == i, 2], c = colors[i+1])

# ax.view_init(0, 0)

# pl.show()
pred_bi_tsne2_sub = pd.DataFrame(bi_tsne.predict(tsne_features[1300:, :]), index = data_df.loc[1300:, 'ID'], columns = ['Class'])

# pred_bi_tsne2_sub = pd.DataFrame(pred_ag_tsne[1300:], index = data_df.loc[1300:, 'ID'], columns = ['Class'])

pred_bi_tsne2_sub.shape
print(pred_bi_tsne2_sub.max())

pred_bi_tsne2_sub.head()
pred_bi_tsne2_sub.replace([0, 1, 3], [3, 0, 1], inplace=True)

pred_bi_tsne2_sub.head()
pred_bi_tsne2_sub = pred_bi_tsne2_sub + 1

pred_bi_tsne2_sub.head()
pred_bi_tsne2_sub.to_csv("Submission.csv")
from IPython.display import HTML 

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"): 

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html) 

create_download_link(pred_bi_tsne2_sub)