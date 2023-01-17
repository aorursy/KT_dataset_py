from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.ticker as ticker



sns.set_style('whitegrid')

%config InlineBackend.figure_format = 'retina'

%matplotlib inline



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))



import Geohash as geo

import matplotlib.ticker as plticker

from datetime import timedelta



## for preprocessing and machine learning

from sklearn.cluster import KMeans, k_means

from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

import sklearn.linear_model as linear_model

from sklearn.preprocessing import MinMaxScaler



## for Deep-learing:

import keras

from keras import models

from keras import layers

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

from keras.layers import LSTM

from keras.layers import Dropout
print(os.listdir('../input'))
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

nRowsRead = 4206321 # specify 'None' if want to read whole file

# training.csv has 4206321 rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('../input/training.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'training.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 6, 15)
from datetime import timedelta

df['day_time'] = df['day'].astype('str') +':'+ df['timestamp']

df['day_time2'] = df['day_time'].apply(lambda x: x.split(':'))

df['day_time3'] = df['day_time2'].apply(lambda x: timedelta(days=int(x[0]),hours=int(x[1]),minutes=int(x[2])))

df['dum_time'] = pd.Timestamp(2019,4,8).normalize() + df['day_time3']

df.drop(['day_time','day_time2','day_time3'],axis=1,inplace=True) # drop irrelevant columns

df.head()



df['time'] = df['dum_time'].dt.time

df['hour'] = df['dum_time'].dt.hour

df['minute'] = df['dum_time'].dt.minute

fig, ax = plt.subplots(figsize=(20,10))

dd_ts = df.groupby('hour')['demand'].sum().reset_index()

sns.lineplot(x='hour', y='demand', data=dd_ts, ax=ax)

loc = plticker.MultipleLocator() # this locator puts ticks at regular intervals

ax.xaxis.set_major_locator(loc)

plt.show()



#time trend shows peak in the morning and trough at 1900-2000hrs

daycycle_dict1 = {}

for i in range(1,8):

    j = i

    while j <= 61:

        daycycle_dict1[j] = i

        j+=7

df['daycycle'] = df['day'].apply(lambda x: daycycle_dict1[x])

fig, ax = plt.subplots(figsize=(20,10))

dd_ts = df.groupby('daycycle')['demand'].sum().reset_index()

sns.lineplot(x='daycycle', y='demand', data=dd_ts, ax=ax)

plt.show()

#Each day of the week has different demand. The 5th & 6th days could be the weekends due to the change in demand behaviour.
df['latitude'] = df['geohash6'].apply(lambda x: geo.decode_exactly(x)[0])

df['longitude'] = df['geohash6'].apply(lambda x: geo.decode_exactly(x)[1])

fig, ax = plt.subplots(figsize=(20,15))

dd_loc = df.groupby(['latitude','longitude'])['demand'].sum().reset_index()

sns.scatterplot(x='longitude', y='latitude', size='demand', sizes=(40, 400), data=dd_loc, ax=ax)

plt.show()

df2 = df.groupby(['geohash6','latitude','longitude'])['demand'].agg(['sum','std']).fillna(0).reset_index() #treat those with nan standard deviation as 0

df2.head()

X = df2.drop('geohash6',axis=1)

Xs  = StandardScaler().fit_transform(X)

Xs  = pd.DataFrame(Xs , columns = X.columns.values)

Xs.head()

def opt_clusters(X, scaling=StandardScaler, k=11):

    #choosing clusters with elbow within cluster sum square errors and silhouette score

    inertia = []

    silh = []

    #standardizing required

    Xs = StandardScaler().fit_transform(X)

    Xs = pd.DataFrame(Xs, columns = X.columns.values)

    for i in range(1,k):

        model = KMeans(n_clusters=i, random_state=0).fit(Xs)

        predicted = model.labels_

        inertia.append(model.inertia_)#low inertia = low cluster sum square error. Low inertia -> Clusters are more compact.

        if i>1:

            silh.append(silhouette_score(Xs, predicted, metric='euclidean')) #High silhouette score = clusters are well separated. The score is based on how much closer data points are to their own clusters (intra-dist) than to the nearest neighbor cluster (inter-dist): (cohesion + separation).  

    plt.plot(np.arange(1, k, step=1), inertia)

    plt.title('Innertia vs clusters')

    plt.xlabel('No. of clusters')

    plt.ylabel('Within Clusters Sum-sq (WCSS)')

    plt.show()

    plt.scatter(np.arange(2, k, step=1), silh)

    plt.title('Sihouette vs clusters')

    plt.xlabel('No. of clusters')

    plt.ylabel('Silhouette score')

    plt.show()

    opt_clusters(Xs, scaling=StandardScaler, k=11)

    #getting prediction and centroids

#select 6 clusters based on silhouette and WCSS

kmeans = KMeans(n_clusters=6, random_state=0).fit(Xs)

predicted = kmeans.labels_

centroids = kmeans.cluster_centers_

Xs['predicted'] = predicted #or X['predicted'] = predicted

df2['cluster'] = Xs['predicted']

fig, ax = plt.subplots(figsize=(20,15))

sns.scatterplot(x='longitude', y='latitude', size='sum', hue='cluster', palette=sns.color_palette("Dark2", 6), sizes=(40, 400), data=df2, ax=ax)

plt.show()

#create a dictionary for these locations

cluster_dict = df2[['geohash6','cluster']].set_index('geohash6')['cluster'].to_dict()

df['cluster'] = df['geohash6'].apply(lambda x: cluster_dict[x])

loc_dict = {0:'clust0', 1:'clust1', 2:'clust2', 3:'clust3', 4:'clust4', 5:'clust5'}

df['cluster'] = df['cluster'].apply(lambda x: loc_dict[x])

df.head()