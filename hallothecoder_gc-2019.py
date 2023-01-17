%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#standard plotly imports

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode



#cufflinks + plotly

import cufflinks as cf

cf.go_offline(connected = True)

# set global

cf.set_config_file(world_readable = True, theme = 'pearl', offline = True)



init_notebook_mode(connected = True)
from scipy.stats import norm
sns.set_style('whitegrid', {'axes,grid' : False})
pd.set_option('display.max_columns', 45)
train_df = pd.read_csv('../input/train_upd.csv')

test_df = pd.read_csv('../input/test_upd.csv')
y_test = pd.read_csv('../input/y_test.csv')
160261 in test_df['cell_name'].values.tolist()
tmp1 = pd.concat((test_df, y_test.drop(columns = 'cell_name')), axis = 1)
train_df.head()
target_name = 'Congestion_Type'
### Check whether all cell towers are different

assert(len(train_df['cell_name'].unique()) == len(train_df))
train_df.drop(columns = ['cell_name', 'par_year', 'par_month']).describe()
print('Length of training data %d and testing data %d'%(train_df.shape[0], test_df.shape[0]))
print('Train - {},  Test - {}'.format(*(train_df.isnull().values.any(), test_df.isnull().values.any())))
fig = plt.figure(figsize = (6, 6))

total = train_df.shape[0]

ax = sns.countplot(x = target_name, data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.1, p.get_height() + 15))
fig = plt.figure(figsize = (6, 6))

total = tmp1.shape[0]

ax = sns.countplot(x = target_name, data = tmp1, order = tmp1[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp1[target_name].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.1, p.get_height() + 15))
plt.figure(figsize = (5, 5))

cnt = train_df[target_name].value_counts()

sizes = cnt.values

plt.pie(sizes, labels = cnt.index, autopct = '%1.1f%%', shadow = True);
col = '4G_rat'

fig = plt.figure(figsize = (6, 6))

total = train_df.shape[0]

ax = sns.countplot(x = col, data = train_df, order = train_df[col].value_counts().sort_values(ascending = False).index)

# ax.set_xticklabels(train_df[col].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.3, p.get_height() + 15))
col = '4G_rat'

fig = plt.figure(figsize = (6, 6))

total = test_df.shape[0]

ax = sns.countplot(x = col, data = test_df, order = test_df[col].value_counts().sort_values(ascending = False).index)

# ax.set_xticklabels(train_df[col].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.3, p.get_height() + 15))
col = 'ran_vendor'

fig = plt.figure(figsize = (6, 6))

total = train_df.shape[0]

ax = sns.countplot(x = col, data = train_df, order = train_df[col].value_counts().sort_values(ascending = False).index)

# ax.set_xticklabels(train_df[col].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.3, p.get_height() + 15))
col = 'ran_vendor'

fig = plt.figure(figsize = (6, 6))

total = test_df.shape[0]

ax = sns.countplot(x = col, data = test_df, order = test_df[col].value_counts().sort_values(ascending = False).index)

# ax.set_xticklabels(train_df[col].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.3, p.get_height() + 15))
### Relation between 4G_rat and Target var

ax = sns.countplot(x = target_name, hue = '4G_rat', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
ax = sns.countplot(x = target_name, hue = 'ran_vendor', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
ax = sns.countplot(x = '4G_rat', hue = 'ran_vendor', data = train_df)
ax = sns.countplot(x = '4G_rat', hue = 'ran_vendor', data = test_df)
sns.boxplot(x = 'tilt', y = 'subscriber_count', hue = 'ran_vendor', data = train_df);
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'tilt', y = 'subscriber_count', hue = 'ran_vendor', data = train_df.sort_values('subscriber_count').iloc[:10000]);
plt.figure(figsize = (20, 5))

sns.boxplot(x = 'tilt', y = 'subscriber_count', hue = 'cell_range', data = train_df.sort_values('subscriber_count').iloc[:10000]);
plt.figure(figsize = (15, 10))

sns.boxplot(x = 'cell_range', y = 'subscriber_count', hue = 'tilt', data = train_df)
plt.figure(figsize = (10, 5))

sns.boxplot(x = 'cell_range', y = 'subscriber_count', hue = 'tilt', data = train_df.sort_values('subscriber_count').loc[:10000])
sns.distplot(train_df['subscriber_count'].values, fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(train_df['subscriber_count'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

plt.ylabel('Frequency')

plt.title('Sub Cnt');
sns.distplot(test_df['subscriber_count'].values, fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(test_df['subscriber_count'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

plt.ylabel('Frequency')

plt.title('Sub Cnt');
ax = sns.boxplot(x = target_name, y = 'subscriber_count', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
train_df[train_df.loc[:, target_name] == 'NC']['subscriber_count'].max()
train_df[train_df.loc[:, target_name] == '4G_BACKHAUL_CONGESTION']['subscriber_count'].max()
fig = plt.figure(figsize = (10, 13))

ax = sns.boxplot(x = target_name, y = 'subscriber_count', hue = '4G_rat', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = 'subscriber_count', hue = 'ran_vendor', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 5))

nc_df = train_df[train_df.loc[:, target_name] == 'NC']

ax = sns.boxplot(x = target_name, y = 'subscriber_count', hue = 'ran_vendor', data = nc_df)

# ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = 'subscriber_count', hue = 'tilt', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = 'subscriber_count', hue = 'cell_range', data = train_df, order = train_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
byte_cols = train_df.columns[8:34]
total_bytes = train_df.loc[:, byte_cols].apply(sum, axis = 1)
tmp = train_df.copy(deep = True)
tmp['total_bytes'] = total_bytes
sns.distplot(tmp['total_bytes'].values, fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(tmp['total_bytes'].values)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

plt.ylabel('Frequency')

plt.title('Sub Cnt');
print("Skewness: %f" % tmp['total_bytes'].skew())

print("Kurtosis: %f" % tmp['total_bytes'].kurt())
print("Skewness: %f" % tmp['subscriber_count'].skew())

print("Kurtosis: %f" % tmp['subscriber_count'].kurt())
y = 'total_bytes'
ax = sns.boxplot(x = target_name, y = y, data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
tmp[tmp.loc[:, target_name] == 'NC'][y].max()
tmp[tmp.loc[:, target_name] == '4G_BACKHAUL_CONGESTION'][y].max()
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = y, hue = '4G_rat', data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = y, hue = 'ran_vendor', data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 5))

nc_df = tmp[tmp.loc[:, target_name] == 'NC']

ax = sns.boxplot(x = target_name, y = y, hue = 'ran_vendor', data = nc_df)

# ax.set_xticklabels(train_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = y, hue = 'tilt', data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = target_name, y = y, hue = 'cell_range', data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
sns.lmplot(x = 'subscriber_count', y = 'total_bytes', data = tmp)
from sklearn.preprocessing import LabelEncoder
def label_encode(df, col):

    lb = LabelEncoder()

    df[col] = lb.fit_transform(df[col])

    return lb
label_encode(tmp, 'ran_vendor');
lb = label_encode(tmp, 'Congestion_Type')
tmp.drop(columns = ['cell_name', 'par_year', 'par_month', target_name], inplace = True)
from sklearn.manifold import TSNE
tsne = TSNE(

    n_components = 2,

    init = 'random',      

    random_state = 101,

    method = 'barnes_hut',

    n_iter = 300,

    verbose = 2,

    angle = 0.5

).fit_transform(tmp[:10000].values)
trace = go.Scatter3d(

    x = tsne[:, 0],

    y = tsne[:, 1],

    z = lb.transform(train_df['Congestion_Type'].values),

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter')
plot_df = pd.DataFrame(dict(x=tsne[:,0], y=tsne[:,1], color=lb.transform(train_df.loc[:9999, target_name])))

sns.lmplot('x', 'y', data=plot_df, hue='color', fit_reg=False, scatter_kws = {'alpha' : 0.3})

plt.show()
tsne = TSNE(

    n_components = 3,

    init = 'random', # pca

    random_state = 101,

    method = 'barnes_hut',

    n_iter = 300,

    verbose = 2,

    angle = 0.5

).fit_transform(tmp[:10000].values)
trace = go.Scatter3d(

    x = tsne[:, 0],

    y = tsne[:, 1],

    z = tsne[:, 2],

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter')
plot_df = pd.DataFrame(dict(x=tsne[:,0], y=tsne[:,1], color=lb.transform(train_df.loc[:9999, target_name])))

sns.lmplot('x', 'y', data=plot_df, hue='color', fit_reg=False, scatter_kws = {'alpha' : 0.3})

plt.show()
import umap
reduce = umap.UMAP(random_state = 223, n_components = 3)  #just for reproducibility

embeddings = reduce.fit_transform(tmp.values)
trace = go.Scatter3d(

    x = embeddings[:, 0],

    y = embeddings[:, 1],

    z = embeddings[:, 2],

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter-umap')
reduce = umap.UMAP(random_state = 446, n_components = 2)  #just for reproducibility

embeddings = reduce.fit_transform(tmp.values)
plot_df = pd.DataFrame(embeddings, columns = ['x', 'y'])

plot_df[target_name] = train_df.loc[:, target_name]
# fig = plt.Figure(figsize = (15, 15))

ax = sns.pairplot(x_vars = ['x'], y_vars = ['y'], data = plot_df, hue = target_name, kind = 'scatter', size = 11, plot_kws = {'s' : 80, 'alpha' : 0.6})

ax.fig.suptitle('Embedding clustered with UMAP');
trace = go.Scatter3d(

    x = embeddings[:, 0],

    y = embeddings[:, 1],

    z = lb.transform(train_df['Congestion_Type'].values),

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter-umap-with-label')
!pip install hdbscan
reduce = umap.UMAP(random_state = 1333, n_components = 2)  #just for reproducibility

embeddings = reduce.fit_transform(tmp.values)
import hdbscan
import time
plot_kwds = {'alpha' : 0.4, 's' : 80, 'linewidths':0}
def plot_clusters(data, algorithm, args, kwds):

    start_time = time.time()

    clusterer = algorithm(*args, **kwds).fit(data)

    labels, strength = hdbscan.approximate_predict(clusterer, data)

    end_time = time.time()

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)

    colors = [sns.desaturate(palette[col], sat) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]

#     colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    plt.scatter(data[:, 0], data[:, 1], c=colors, **plot_kwds)

    frame = plt.gca()

    frame.axes.get_xaxis().set_visible(False)

    frame.axes.get_yaxis().set_visible(False)

    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)

    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

    return clusterer
clusterer = plot_clusters(embeddings, hdbscan.HDBSCAN, (), {'prediction_data':True, 'min_cluster_size':15})
np.unique(clusterer.labels_)
clusterer = plot_clusters(tsne, hdbscan.HDBSCAN, (), {'prediction_data':True, 'min_cluster_size':15})
np.unique(clusterer.labels_)
tmp_copy = tmp.copy(deep = True)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
def normalize_col(normalize, df, col):

    if normalize == 1:

        scaler = MinMaxScaler(feature_range = (0, 1))

        df[col] = scaler.fit_transform(np.expand_dims(df[col].values, axis = 1))

    

    elif normalize == 2:

        scaler = StandardScaler()

        df[col] = scaler.fit_transform(np.expand_dims(df[col].values, axis = 1))

    

    elif normalize == 3:

        scaler = RobustScaler()

        df[col] = scaler.fit_transform(np.expand_dims(df[col].values, axis = 1))
cols2normalize = list(tmp_copy.columns[4:32]) + [tmp_copy.columns[-1]]
for col in cols2normalize:

    normalize_col(3, tmp_copy, col)
tmp_copy.describe()
tsne = TSNE(

    n_components = 3,

    init = 'random', # pca

    random_state = 101,

    method = 'barnes_hut',

    n_iter = 300,

    verbose = 2,

    angle = 0.5

).fit_transform(tmp_copy[:10000].values)
trace = go.Scatter3d(

    x = tsne[:, 0],

    y = tsne[:, 1],

    z = tsne[:, 2],

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter')
plot_df = pd.DataFrame(dict(x=tsne[:,0], y=tsne[:,1], color=lb.transform(train_df.loc[:9999, target_name])))

sns.lmplot('x', 'y', data=plot_df, hue='color', fit_reg=False, scatter_kws = {'alpha' : 0.3})

plt.show()
tsne = TSNE(

    n_components = 2,

    init = 'random',      

    random_state = 101,

    method = 'barnes_hut',

    n_iter = 300,

    verbose = 2,

    angle = 0.5

).fit_transform(tmp_copy[:10000].values)
trace = go.Scatter3d(

    x = tsne[:, 0],

    y = tsne[:, 1],

    z = lb.transform(train_df['Congestion_Type'].values),

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter')
plot_df = pd.DataFrame(dict(x=tsne[:,0], y=tsne[:,1], color=lb.transform(train_df.loc[:9999, target_name])))

sns.lmplot('x', 'y', data=plot_df, hue='color', fit_reg=False, scatter_kws = {'alpha' : 0.3})

plt.show()
import umap
reduce = umap.UMAP(random_state = 223, n_components = 3)  #just for reproducibility

embeddings = reduce.fit_transform(tmp_copy.values)
trace = go.Scatter3d(

    x = embeddings[:, 0],

    y = embeddings[:, 1],

    z = embeddings[:, 2],

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter-umap')
reduce = umap.UMAP(random_state = 446, n_components = 2)  #just for reproducibility

embeddings = reduce.fit_transform(tmp_copy.values)
plot_df = pd.DataFrame(embeddings, columns = ['x', 'y'])

plot_df[target_name] = train_df.loc[:, target_name]
# fig = plt.Figure(figsize = (15, 15))

ax = sns.pairplot(x_vars = ['x'], y_vars = ['y'], data = plot_df, hue = target_name, kind = 'scatter', size = 11, plot_kws = {'s' : 80, 'alpha' : 0.6})

ax.fig.suptitle('Embedding clustered with UMAP');
trace = go.Scatter3d(

    x = embeddings[:, 0],

    y = embeddings[:, 1],

    z = lb.transform(train_df['Congestion_Type'].values),

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter-umap-with-label')
df = train_df.copy(deep = True)
from datetime import datetime, date
df.insert(7, 'day', np.nan)

df.head()
df['day'] = df.apply(lambda x : datetime.strptime("%s|%s|%s"%(x['par_day'], x['par_month'], x['par_year']), "%d|%m|%Y").date().weekday(), axis = 1)
df.head()
byte_cols = df.columns[9:35]
df['total_bytes'] = df.loc[:, byte_cols].apply(sum, axis = 1)
fig = plt.figure(figsize = (10, 10))

ax = sns.boxplot(x = 'day', y = 'total_bytes', hue = target_name, data = df)#, order = df[].value_counts().sort_values(ascending = False).index)

# ax.set_xticklabels(df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
fig = plt.figure(figsize = (6, 6))

total = df.shape[0]

ax = sns.barplot(x = 'day', y = 'total_bytes', data = df)

# ax.set_xticklabels(train_df[col].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.1, p.get_height() + 15))
tmp_drop_date = tmp.copy(deep = True)
tmp_drop_date.drop(columns = ['par_day', 'par_hour', 'par_min'], inplace = True)
tsne = TSNE(

    n_components = 2,

    init = 'random',      

    random_state = 101,

    method = 'barnes_hut',

    n_iter = 300,

    verbose = 2,

    angle = 0.5

).fit_transform(tmp_drop_date[:10000].values)
trace = go.Scatter3d(

    x = tsne[:, 0],

    y = tsne[:, 1],

    z = lb.transform(train_df['Congestion_Type'].values),

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter')
plot_df = pd.DataFrame(dict(x=tsne[:,0], y=tsne[:,1], color=lb.transform(train_df.loc[:9999, target_name])))

sns.lmplot('x', 'y', data=plot_df, hue='color', fit_reg=False, scatter_kws = {'alpha' : 0.3})

plt.show()
tsne = TSNE(

    n_components = 3,

    init = 'random', # pca

    random_state = 101,

    method = 'barnes_hut',

    n_iter = 300,

    verbose = 2,

    angle = 0.5

).fit_transform(tmp_drop_date[:10000].values)
trace = go.Scatter3d(

    x = tsne[:, 0],

    y = tsne[:, 1],

    z = tsne[:, 2],

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter')
plot_df = pd.DataFrame(dict(x=tsne[:,0], y=tsne[:,1], color=lb.transform(train_df.loc[:9999, target_name])))

sns.lmplot('x', 'y', data=plot_df, hue='color', fit_reg=False, scatter_kws = {'alpha' : 0.3})

plt.show()
import umap
reduce = umap.UMAP(random_state = 223, n_components = 3)  #just for reproducibility

embeddings = reduce.fit_transform(tmp_drop_date.values)
trace = go.Scatter3d(

    x = embeddings[:, 0],

    y = embeddings[:, 1],

    z = embeddings[:, 2],

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter-umap')
reduce = umap.UMAP(random_state = 446, n_components = 2)  #just for reproducibility

embeddings = reduce.fit_transform(tmp_drop_date.values)
plot_df = pd.DataFrame(embeddings, columns = ['x', 'y'])

plot_df[target_name] = train_df.loc[:, target_name]
# fig = plt.Figure(figsize = (15, 15))

ax = sns.pairplot(x_vars = ['x'], y_vars = ['y'], data = plot_df, hue = target_name, kind = 'scatter', size = 11, plot_kws = {'s' : 80, 'alpha' : 0.6})

ax.fig.suptitle('Embedding clustered with UMAP');
trace = go.Scatter3d(

    x = embeddings[:, 0],

    y = embeddings[:, 1],

    z = lb.transform(train_df['Congestion_Type'].values),

    mode = 'markers',

    marker = dict(

        size = 12,

        color = lb.transform(train_df['Congestion_Type'].values),

        colorscale = 'Portland',

        colorbar = dict(title = 'Congestion'),

        line = dict(color='rgb(255, 255, 255)'),

        opacity = 0.5

    )

)



layout = go.Layout(

    scene = dict(

        camera = dict(

            eye = dict(

            x = 0.5,

            y = 0.5,

            z = 0.5

            )

        )

    ),

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

fig = go.Figure(data = [trace], layout = layout)

iplot(fig, filename = '3d-scatter-umap-with-label')
reduce = umap.UMAP(random_state = 1333, n_components = 2)  #just for reproducibility

embeddings = reduce.fit_transform(tmp_drop_date.values)
plot_kwds = {'alpha' : 0.4, 's' : 80, 'linewidths':0}
def plot_clusters(data, algorithm, args, kwds):

    start_time = time.time()

    clusterer = algorithm(*args, **kwds).fit(data)

    labels, strength = hdbscan.approximate_predict(clusterer, data)

    end_time = time.time()

    palette = sns.color_palette('deep', np.unique(labels).max() + 1)

    colors = [sns.desaturate(palette[col], sat) for col, sat in zip(clusterer.labels_, clusterer.probabilities_)]

#     colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

    plt.scatter(data[:, 0], data[:, 1], c=colors, **plot_kwds)

    frame = plt.gca()

    frame.axes.get_xaxis().set_visible(False)

    frame.axes.get_yaxis().set_visible(False)

    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)

    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

    return clusterer
clusterer = plot_clusters(embeddings, hdbscan.HDBSCAN, (), {'prediction_data':True, 'min_cluster_size':15})
np.unique(clusterer.labels_)
clusterer = plot_clusters(tsne, hdbscan.HDBSCAN, (), {'prediction_data':True, 'min_cluster_size':15})
np.unique(clusterer.labels_)
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,cross_validate

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from sklearn.linear_model import SGDClassifier,Perceptron,PassiveAggressiveClassifier,RidgeClassifier, LogisticRegression, LogisticRegressionCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_selection import SelectFromModel,SelectKBest,chi2

from sklearn.svm import LinearSVC, SVC

from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier,NearestCentroid

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.extmath import density

from sklearn import metrics
# for col in tmp.columns:

#     normalize_col(3, tmp, col)



# for col in tmp_drop_date.columns:

#     normalize_col(3, tmp_drop_date, col)
tmp_cols = tmp.columns[5 : -5].values.tolist() + [tmp.columns[-1]]
tmp_date_cols = tmp_drop_date.columns[2 : -5].values.tolist() + [tmp_drop_date.columns[-1]]
assert(tmp_cols == tmp_date_cols)
for col in tmp_cols:

    normalize_col(3, tmp, col)



for col in tmp_date_cols:

    normalize_col(3, tmp_drop_date, col)
X = tmp.values

X_date = tmp_drop_date.values
y = lb.transform(train_df['Congestion_Type'])

y
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 4242, stratify = y)

x_date_train, x_date_valid, y_train, y_valid = train_test_split(X_date, y, test_size = 0.2, random_state = 2424, stratify = y)
target_names = lb.classes_

target_names
from time import time
def benchmark(clf, name = None, x_train = x_train, x_valid = x_valid):

    print('_' * 80)

    print("Training: ")

    print(clf)

    t0 = time()

    clf.fit(x_train, y_train)

    train_time = time() - t0

    print("train time: %0.3fs" % train_time)



    t0 = time()

    pred = clf.predict(x_valid)

    test_time = time() - t0

    print("test time:  %0.3fs" % test_time)

    

    score = metrics.accuracy_score(y_valid, pred)

    

    train_pred = clf.predict(x_train)

    train_score = metrics.accuracy_score(y_train, train_pred)

    

    print("Train accuracy:   %0.3f" % train_score)

    print("Validation accuracy:   %0.3f" % score)



    

    print("classification report:")

    print(metrics.classification_report(y_valid, pred,

                                            target_names=target_names))



    print("confusion matrix:")

    cm = metrics.confusion_matrix(y_valid, pred)

    print(cm)

    fig = plt.figure(figsize=(18,20))

    ax = sns.heatmap(cm, annot=True, annot_kws={"size": 14},

            fmt='g', cmap='OrRd', xticklabels=target_names, yticklabels=target_names)

    txt = name + ' method has an accurcy of ' + str(100*score)

    fig.text(.5, .05, txt, ha='center',fontsize='xx-large')

    plt.tight_layout()

#     plt.savefig('./plots/' + name + '.jpg')



    print()

    clf_descr = str(clf).split('(')[0]

    return clf_descr, score, train_time, test_time
def trim(s):

    """Trim string to fit on terminal (assuming 80-column display)"""

    return s if len(s) <= 80 else s[:77] + "..."
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000), "LR"),

#         (SVC(), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_train, x_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,

                                       tol=1e-4),'LinearSVC',x_train, x_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,

                                           penalty=penalty),'SGDClassifer_'+penalty,x_train, x_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_train, x_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_train, x_valid))
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000), "LR_LBFGS"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'saga', multi_class = 'multinomial', max_iter = 1000, penalty = 'l1'), "LR_SAGA"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'newton-cg', multi_class = 'multinomial', max_iter = 1000), "LR_NEWTON-CG"),

        (SVC(gamma = 'auto'), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_date_train, x_date_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,

                                       tol=1e-4),'LinearSVC',x_date_train, x_date_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,

                                           penalty=penalty),'SGDClassifer_'+penalty,x_date_train, x_date_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_date_train, x_date_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter = 3000,

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_date_train, x_date_valid))
from sklearn.decomposition import PCA
pca = PCA(n_components = 20)

X = pca.fit_transform(tmp.values)
pca.explained_variance_ratio_.sum()
pca = PCA(n_components = 20)

X_date = pca.fit_transform(tmp_drop_date.values)
pca.explained_variance_ratio_.sum()
X = np.concatenate((tmp.values, X), axis = 1)
X_date = np.concatenate((tmp_drop_date.values, X_date), axis = 1)
X.shape, X_date.shape, y.shape
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 4242, stratify = y)

x_date_train, x_date_valid, y_train, y_valid = train_test_split(X_date, y, test_size = 0.2, random_state = 2424, stratify = y)
target_names = lb.classes_

target_names
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000), "LR"),

#         (SVC(), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_train, x_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,

                                       tol=1e-4),'LinearSVC',x_train, x_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,

                                           penalty=penalty),'SGDClassifer_'+penalty,x_train, x_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_train, x_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_train, x_valid))
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000), "LR_LBFGS"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'saga', multi_class = 'multinomial', max_iter = 1000, penalty = 'l1'), "LR_SAGA"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'newton-cg', multi_class = 'multinomial', max_iter = 1000), "LR_NEWTON-CG"),

        (SVC(gamma = 'auto'), "SVC"),

#         (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300), "Ridge Classifier"),

#         (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1), "Perceptron"),

#         (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_date_train, x_date_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,

                                       tol=1e-3),'LinearSVC',x_date_train, x_date_valid))



    # Train SGD model

#     results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,

#                                            penalty=penalty),'SGDClassifer_'+penalty,x_date_train, x_date_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_date_train, x_date_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter = 3000,

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_date_train, x_date_valid))
tmp_bytes = tmp.copy(deep = True)

tmp_drop_date_bytes = tmp_drop_date.copy(deep = True)
tmp_cols_bytes = tmp_bytes.columns[5 : -5].values.tolist() + [tmp_bytes.columns[-1]]
tmp_drop_date_cols_bytes = tmp_drop_date_bytes.columns[2 : -5].values.tolist() + [tmp_drop_date_bytes.columns[-1]]
assert(tmp_cols_bytes == tmp_drop_date_cols_bytes)
tmp_cols_bytes.__len__()
pca = PCA(n_components = 20)

X_tmp_bytes = pca.fit_transform(tmp_bytes.loc[:, tmp_cols_bytes].values)
pca.explained_variance_ratio_.sum()
pca = PCA(n_components = 20)

X_tmp_date_bytes = pca.fit_transform(tmp_drop_date_bytes.loc[:, tmp_drop_date_cols_bytes].values)
pca.explained_variance_ratio_.sum()
tmp_bytes.drop(columns = tmp_cols_bytes, inplace = True)

tmp_drop_date_bytes.drop(columns = tmp_drop_date_cols_bytes, inplace = True)
tmp_bytes.head()
tmp_drop_date_bytes.head()
X = np.concatenate((tmp_bytes.values, X_tmp_bytes), axis = 1)
X_date = np.concatenate((tmp_drop_date_bytes.values, X_tmp_date_bytes), axis = 1)
y = lb.transform(train_df['Congestion_Type'])

y
X.shape, X_date.shape, y.shape
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 4242, stratify = y)

x_date_train, x_date_valid, y_train, y_valid = train_test_split(X_date, y, test_size = 0.2, random_state = 2424, stratify = y)
target_names = lb.classes_

target_names
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000), "LR"),

#         (SVC(), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_train, x_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,

                                       tol=1e-4),'LinearSVC',x_train, x_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,

                                           penalty=penalty),'SGDClassifer_'+penalty,x_train, x_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_train, x_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_train, x_valid))
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000), "LR_LBFGS"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'saga', multi_class = 'multinomial', max_iter = 1000, penalty = 'l1'), "LR_SAGA"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'newton-cg', multi_class = 'multinomial', max_iter = 1000), "LR_NEWTON-CG"),

        (SVC(gamma = 'auto'), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_date_train, x_date_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,

                                       tol=1e-4),'LinearSVC',x_date_train, x_date_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,

                                           penalty=penalty),'SGDClassifer_'+penalty,x_date_train, x_date_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_date_train, x_date_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter = 3000,

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_date_train, x_date_valid))
byte_cols = train_df.columns[8:34]
total_bytes = train_df.loc[:, byte_cols].apply(sum, axis = 1)
tmp = train_df.copy(deep = True)
tmp['total_bytes'] = total_bytes
total_bytes = tmp1.loc[:, byte_cols].apply(sum, axis = 1)
tmp1['total_bytes'] = total_bytes
sns.distplot(tmp['total_bytes'].values, fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(tmp['total_bytes'].values)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

plt.ylabel('Frequency')

plt.title('Sub Cnt');
sns.distplot(tmp1['total_bytes'].values, fit = norm)



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(tmp1['total_bytes'].values)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

plt.ylabel('Frequency')

plt.title('Sub Cnt');
print("Skewness: %f" % tmp['total_bytes'].skew())

print("Kurtosis: %f" % tmp['total_bytes'].kurt())
print("Skewness: %f" % tmp['subscriber_count'].skew())

print("Kurtosis: %f" % tmp['subscriber_count'].kurt())
print("Skewness: %f" % tmp1['total_bytes'].skew())

print("Kurtosis: %f" % tmp1['total_bytes'].kurt())
print("Skewness: %f" % tmp1['subscriber_count'].skew())

print("Kurtosis: %f" % tmp1['subscriber_count'].kurt())
ax = sns.boxplot(x = target_name, y = 'subscriber_count', data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
tmp[tmp.loc[:, target_name] == 'NC']['subscriber_count'].max()
tmp[tmp.loc[:, target_name] == '4G_BACKHAUL_CONGESTION']['subscriber_count'].max()
x = tmp[tmp.loc[:, target_name] == 'NC']

x[x.loc[:, 'subscriber_count'] < 4000]['total_bytes'].max()
x = tmp[tmp.loc[:, target_name] == 'NC']

x[x.loc[:, 'total_bytes'] < 12000]['subscriber_count'].max()
ax = sns.boxplot(x = target_name, y = 'subscriber_count', data = tmp1, order = tmp1[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp1[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
tmp1[tmp1.loc[:, target_name] == 'NC']['subscriber_count'].max()
tmp1[tmp1.loc[:, target_name] == '4G_BACKHAUL_CONGESTION']['subscriber_count'].max()
y = 'total_bytes'
ax = sns.boxplot(x = target_name, y = y, data = tmp, order = tmp[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
tmp[tmp.loc[:, target_name] == 'NC'][y].max()
tmp[tmp.loc[:, target_name] == '4G_BACKHAUL_CONGESTION'][y].max()
ax = sns.boxplot(x = target_name, y = y, data = tmp1, order = tmp1[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(tmp1[target_name].value_counts().sort_values(ascending = False).index, rotation = 90);
tmp1[tmp1.loc[:, target_name] == 'NC'][y].max()
tmp1[tmp1.loc[:, target_name] == '4G_BACKHAUL_CONGESTION'][y].max()
all_df = tmp[tmp.loc[:, ['subscriber_count', 'total_bytes']].apply(lambda x : x['subscriber_count'] < 3000 and x['total_bytes'] < 110000, axis = 1)]
all_df.__len__()
all_df.reset_index(drop = True, inplace = True)
len(tmp[tmp.loc[:, 'subscriber_count'] < 4000])
len(tmp[tmp.loc[:, 'total_bytes'] < 120000])
assert(all_df[all_df.loc[:, target_name] == 'NC'].__len__() == train_df[train_df.loc[:, target_name] == 'NC'].__len__())
three_df = tmp[tmp.loc[:, ['subscriber_count', 'total_bytes']].apply(lambda x : x['subscriber_count'] >= 3000 or x['total_bytes'] >= 110000, axis = 1)]
assert(three_df[three_df.loc[:, target_name] != 'NC'].__len__() == three_df[three_df.loc[:, target_name] != 'NC'].__len__())
fig = plt.figure(figsize = (6, 6))

total = all_df.shape[0]

ax = sns.countplot(x = target_name, data = all_df, order = all_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(all_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.1, p.get_height() + 15))
all_df['labels'] = all_df[target_name].apply(lambda x : 0 if x == 'NC' else 1)
fig = plt.figure(figsize = (6, 6))

total = all_df.shape[0]

ax = sns.countplot(x = 'labels', data = all_df, order = all_df['labels'].value_counts().sort_values(ascending = False).index)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.3, p.get_height() + 15))
bin_df = all_df.drop(columns = ['cell_name', 'par_year', 'par_month', 'par_day', 'par_hour', 'par_min', 'Congestion_Type'])

bin_df.head()
from sklearn.preprocessing import LabelEncoder
def label_encode(df, col):

    lb = LabelEncoder()

    df[col] = lb.fit_transform(df[col])

    return lb
label_encode(bin_df, 'ran_vendor');
for col in bin_df.columns:

    normalize_col(3, bin_df, col)
X = bin_df.drop(columns = 'labels').values

y = bin_df['labels']
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 4242, stratify = y)
target_names = ['NC', 'CONGESTION']
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000, class_weight='balanced'), "LR_LBFGS"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'saga', multi_class = 'multinomial', max_iter = 1000, penalty = 'l1',class_weight='balanced'), "LR_SAGA"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'newton-cg', multi_class = 'multinomial', max_iter = 1000,class_weight='balanced'), "LR_NEWTON-CG"),

        (SVC(gamma = 'scale',class_weight='balanced'), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300,class_weight='balanced'), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1,class_weight='balanced'), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1, class_weight='balanced'), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance'), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1,class_weight='balanced'), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_train, x_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,class_weight='balanced',

                                       tol=1e-4),'LinearSVC',x_train, x_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,class_weight='balanced',

                                           penalty=penalty),'SGDClassifer_'+penalty,x_train, x_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,class_weight='balanced',

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_train, x_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter = 3000,class_weight='balanced',

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_train, x_valid))
mul_df = all_df.drop(columns = ['cell_name', 'par_year', 'par_month', 'par_day', 'par_hour', 'par_min'])
label_encode(mul_df, 'ran_vendor');
lb = label_encode(mul_df, 'Congestion_Type')
X = mul_df.drop(columns = 'Congestion_Type')

y = mul_df['Congestion_Type'].values
for col in X.columns:

    normalize_col(3, X, col)
X = X.values
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 4242, stratify = y)
target_names = lb.classes_

target_names
results = []

for clf, name in (

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 1000, class_weight='balanced'), "LR_LBFGS"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'saga', multi_class = 'multinomial', max_iter = 1000, penalty = 'l1',class_weight='balanced'), "LR_SAGA"),

        (LogisticRegression(C = 2.0, tol = 1e-5, random_state = 420, solver = 'newton-cg', multi_class = 'multinomial', max_iter = 1000,class_weight='balanced'), "LR_NEWTON-CG"),

        (SVC(gamma = 'scale',class_weight='balanced'), "SVC"),

        (RidgeClassifier(tol=1e-2, solver="lsqr", max_iter=300,class_weight='balanced'), "Ridge Classifier"),

        (Perceptron(max_iter=300, tol=1e-3, n_jobs=-1,class_weight='balanced'), "Perceptron"),

        (PassiveAggressiveClassifier(max_iter=2000, tol=1e-5, n_jobs=-1, class_weight='balanced'), "Passive-Aggressive"),

        (KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance'), "kNN"),

        (RandomForestClassifier(n_estimators=100, n_jobs = -1,class_weight='balanced'), "Random forest")):

    print('=' * 80)

    print(name)

    results.append(benchmark(clf,name,x_train, x_valid))



for penalty in ["l2", "l1"]:

    print('=' * 80)

    print("%s penalty" % penalty.upper())

    # Train Liblinear model

    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,max_iter=3000,class_weight='balanced',

                                       tol=1e-4),'LinearSVC',x_train, x_valid))



    # Train SGD model

    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=2000, tol=1e-4,class_weight='balanced',

                                           penalty=penalty),'SGDClassifer_'+penalty,x_train, x_valid))



# Train SGD with Elastic Net penalty

print('=' * 80)

print("Elastic-Net penalty")

results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=1000, tol=1e-4,class_weight='balanced',

                                       penalty="elasticnet"),'SGDClassifer_elasticnet',x_train, x_valid))



print('=' * 80)

print("LinearSVC with L1-based feature selection")

# The smaller C, the stronger the regularization.

# The more regularization, the more sparsity.

results.append(benchmark(Pipeline([

  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter = 3000,class_weight='balanced',

                                                  tol=1e-4, ))),

  ('classification', LinearSVC(penalty="l2", class_weight = 'balanced'))]),'pipeline',x_train, x_valid))
fig = plt.figure(figsize = (6, 6))

total = three_df.shape[0]

ax = sns.countplot(x = target_name, data = three_df, order = three_df[target_name].value_counts().sort_values(ascending = False).index)

ax.set_xticklabels(three_df[target_name].value_counts().sort_values(ascending = False).index, rotation = 90)

for p in ax.patches:

    ax.annotate('{:.2f}%'.format(p.get_height() * 100 / total), (p.get_x() + 0.1, p.get_height() + 15))
# no_cgs_df = train_df[train_df.loc[:, target_name] == 'NC']

# cgs_df = train_df[train_df.loc[:, target_name] != 'NC']