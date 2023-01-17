# ****-------Notebook Summary----***

#Data Science, Machine Learning,Deep Learning(Artificial Neural Networks(ANN))

#Data Visualization,EDA Analysis, Data Pre-processing,Data Manipulation,Data Cleaning
#--------------------------------------------------------

##(UnSupervised Machine Learning Algorithm)

#Part1=K-means Clustering or Partition clustering

#Part2=Hierarchical Clustering or Agglomerative clustering.

#Part3 =K-means Clustering or Partition clustering for Multiple Cluster

#---------------
#find This Note Book:https://www.kaggle.com/sohelranaccselab/market-customer-segmentation-using-unsupervised-ml?scriptVersionId=42067029

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
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Data Splitting
#Data Read
file_path = '../input/customer-segmentation-tutorial-in-python'
df=pd.read_csv(f'{file_path}/Mall_Customers.csv')
df.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes('object').columns:
    df[col] = le.fit_transform(df[col])
df.head()
df = df.loc[:,~df.columns.duplicated()]
import pandas_profiling
# preparing profile report

profile_report = pandas_profiling.ProfileReport(df,minimal=True)
profile_report
df.info()
df.shape
df.apply(lambda x: sum(x.isnull()),axis=0)
df.groupby("Gender").mean()
df.groupby("Age").mean()
import seaborn; seaborn.set()
df.plot();
df.corr()
def correlation_matrix(d):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Market Customer Segmentation  features correlation\n',fontsize=15)
    labels=df.columns
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=9)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])
    plt.show()

correlation_matrix(df)
#Plotting data 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True,linecolor='red',linewidths=3,cmap = 'plasma')
f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(df.corr(),annot=True, linewidths=.5,fmt='.1f',ax=ax)
sns.pairplot(df,diag_kind="kde")
plt.show()
i=1
plt.figure(figsize=(25,20))
for c in df.describe().columns[:]:
    plt.subplot(5,3,i)
    plt.title(f"Histogram of {c}",fontsize=10)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.hist(df[c],bins=20,color='green',edgecolor='k')
    i+=1
plt.show()
#checking the target variable countplot
sns.countplot(data=df,x = 'Gender',palette='plasma')
i=1
plt.figure(figsize=(35,25))
for c in df.columns[2:]:
    plt.subplot(2,3,i)
    plt.title(f"Boxplot of {c}",fontsize=16)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    sns.boxplot(y=df[c],x=df['Gender'])
    i+=1
plt.show()
#Numerical Columns data distribution
sns.set()
fig = plt.figure(figsize = [15,20])
cols = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
cnt = 1
for col in cols :
    plt.subplot(2,3,cnt)
    sns.distplot(df[col],hist_kws=dict(edgecolor="k", linewidth=1,color='green'),color='red')
    cnt+=1
plt.show() 
# Distplot
fig, ax2 = plt.subplots(2, 3, figsize=(16, 16))
sns.distplot(df['Gender'],ax=ax2[0][0])
sns.distplot(df['Age'],ax=ax2[0][1])
sns.distplot(df['Annual Income (k$)'],ax=ax2[0][2])
sns.distplot(df['Spending Score (1-100)'],ax=ax2[1][1])

sns.set()
fig = plt.figure(figsize = [15,20])
cols = ['Age', 'Spending Score (1-100)', 'Annual Income (k$)']
cnt = 1
for col in cols :
    plt.subplot(4,3,cnt)
    sns.violinplot(x="Gender", y=col, data=df)
    cnt+=1
plt.show()
#data Pre-Processing:Additinal 
class pre_processing:
    
    def __init__(self, data):
        self.data   = data
    
    def missing_percent_plot(self):
        missing_col = list(self.data.isna().sum() != 0)

        try:
            if True not in missing_col:
                raise ValueError("There is no missing values.")

            self.data = self.data.loc[:,missing_col]
            missing_percent = (self.data.isna().sum()/ self.data.shape[0]) * 100

            df = pd.DataFrame()
            df['Total']        = self.data.isna().sum()
            df['perc_missing'] = missing_percent
            p = sns.barplot(x=df.perc_missing.index, y='perc_missing', data=df); plt.xticks(rotation=90)
            plt.xticks(rotation=45);p.tick_params(labelsize=14)
        except:
            return print('There is no missing values...')
        return df.sort_values(ascending =False, by='Total', axis =0)
    
    def reduce_mem_usage(self, verbose=True):
    
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = self.data.memory_usage().sum() / 1024**2 # Memory total(Ram)

        for col in tqdm(self.data.columns):
            col_type = self.data[col].dtypes
            
            if col_type in numerics:
                c_min = self.data[col].min()
                c_max = self.data[col].max()

                # Int
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.data[col] = self.data[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.data[col] = self.data[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.data[col] = self.data[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.data[col] = self.data[col].astype(np.int64)  

                # Float
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.data[col] = self.data[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.data[col] = self.data[col].astype(np.float32)
                    else:
                        self.data[col] = self.data[col].astype(np.float64)

        end_mem = self.data.memory_usage().sum() / 1024**2
        if verbose: 
            print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return self.data
    
    def value_symmetry(self, target):
        return self.data[target].value_counts().plot('bar')
    
    def kde_plots(self, columns : list, hue_col : str):
        
        
        for c in columns:
            # hue loop
            for hue_value in self.data[hue_col].unique():
                sns.distplot(self.data[self.data[hue_col] == hue_value][c], hist = False, label=hue_value)
            plt.show()
    
    def plots(self, columns : list, hue_col):
        _, axs = plt.subplots(int(round(len(columns) / 2, 0)), 5,figsize=(12,12))
        
        for n, c in enumerate(columns):
            # hue loop
            for hue_value in self.data[hue_col].unique():
                sns.distplot(self.data[self.data[hue_col] == hue_value][c], hist = False, label=hue_value, ax=axs[n//5][n%5])
            plt.tight_layout()
        plt.show()
            
df1 = pre_processing(df)
columns=['Annual Income (k$)','Spending Score (1-100)', 'Age']
hue_col = 'Gender'
df1.plots(columns, hue_col)
df1.missing_percent_plot()
df.head()
#Un-Supervised machine Learning Models Performance
origin = df.copy()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

np.random.seed(5)
"""What is clustering?¶
Clustering is an unsupervized learning technique where you take the entire dataset and find the "groups of similar entities" within the dataset. Hence there is no labels within the dataset.

Useful for organizing very large dataset into meaningful clusters that can be useful and actions can be taken upon. For example, take entire customer base of more than 1M records and try to group into high value customers, low value customers and so on.

What questions does clustering typically tend to answer?

Types of pages are there on the Web?
Types of customers are there in my market?
Types of people are there on a Social network?
Types of E-mails in my Inbox?
Types of Genes the human genome has?
From clustering to classification
Clustering is base of all the classification problems. Initially, say we have a large ungrouped number of users in a new social media platform. We know for certain that the number of users will not be equal to the number of groups in the social media, and it will be reasonably finite.
Even though each user can vary in fine-grain, they can be reasonably grouped into clusters.
Each of these grouped clusters become classes when we know what group each of these users fall into.

"""

#Partition clustering
standard_scalar = StandardScaler()
data_scaled = standard_scalar.fit_transform(df)
df = pd.DataFrame(data_scaled, columns=df.columns)
df.head()
from sklearn.cluster import KMeans

km = KMeans(init="random", n_clusters=5)
km.fit(df)
km.labels_
km.cluster_centers_
# k-means determine k
distortions = []
K = range(1, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df)
    distortions.append(kmeanModel.inertia_)
    
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('No of clusters (k)')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
estimators = [('k_means_5', KMeans(n_clusters=5, init='k-means++')),
              ('k_means_2', KMeans(n_clusters=2, init='k-means++')),
              ('k_means_bad_init', KMeans(n_clusters=2, n_init=1, init='random'))]

fignum = 1
titles = ['5 clusters', '2 clusters', '2 clusters, bad initialization']

for name, est in estimators:
    fig = plt.figure(fignum, figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(df)
    labels = est.labels_

    ax.scatter(df.values[:, 3], df.values[:, 0], df.values[:, 2], c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('CustomerID')
    ax.set_ylabel('Gender')
    ax.set_zlabel('Age')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

#Hierarchical Clustering or Agglomerative clustering.
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering().fit(df)
clustering
clustering.labels_
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(df)

plt.figure(fignum, figsize=(10, 6))
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

#Final Part
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
ss = StandardScaler()
Mall_Customers = pd.DataFrame(ss.fit_transform(df), columns=df.columns)
inertia_list = []
for n_clusters in range(1, 20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(Mall_Customers)
    inertia_list.append(kmeans.inertia_)
sns.lineplot(x= [i for i in range(1, 20)], y=inertia_list,marker=True)
kmeans = KMeans(n_clusters=4, random_state=42).fit(Mall_Customers)
new_Mall_Customers = pd.concat([pd.DataFrame(kmeans.labels_, columns=['labels']), Mall_Customers], axis=1)
from sklearn.decomposition import PCA
pca=PCA(n_components=4)
pca.fit(Mall_Customers)
pca.explained_variance_ratio_.sum()
pca_Mall_Customers = pd.DataFrame(pca.fit_transform(Mall_Customers))
new_Mall_Customers['labels'] = kmeans.labels_
origin['labels']             = kmeans.labels_
# Multi-dimmention visualization with standardized and pca applied data
pd.plotting.parallel_coordinates(new_Mall_Customers, 'labels', color=('#556270', '#C7F464', '#FF6B6B', '#000000'))
# plot with raw data
pd.plotting.parallel_coordinates(origin, 'labels', color=('#556270', '#C7F464', '#FF6B6B', '#000000'))
# plot with standardized data
ss_origin = pd.DataFrame(ss.fit_transform(Mall_Customers), columns=Mall_Customers.columns)
ss_origin['labels'] = kmeans.labels_
pd.plotting.parallel_coordinates(ss_origin, 'labels', color=('#556270', '#C7F464', '#FF6B6B', '#000000'))
#Lets examine raw data also

inertia_list = []
for n_clusters in range(1, 20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(origin)
    inertia_list.append(kmeans.inertia_)
# with raw data
sns.lineplot(x= [i for i in range(1, 20)], y=inertia_list,marker=True)
del origin['labels']
kmeans = KMeans(n_clusters=2, random_state=42).fit(origin)
origin['labels'] = kmeans.labels_
pd.plotting.parallel_coordinates(origin, 'labels', color=('#556270', '#C7F464'))
#Raw data with two clusters looks more clear in Above picture
del origin['labels']
pca=PCA(n_components=3)
pca.fit(origin)
pca.explained_variance_ratio_.sum()
origin_3d_pca = pd.DataFrame(pca.fit_transform(origin))
inertia_list = []
for n_clusters in range(1, 20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(origin_3d_pca)
    inertia_list.append(kmeans.inertia_)
    
# with raw data
sns.lineplot(x= [i for i in range(1, 20)], y=inertia_list,marker=True)
#Let's visualize 3 dimensional scatter plot
kmeans = KMeans(n_clusters=2, random_state=42).fit(origin_3d_pca)
origin_3d_pca['labels'] = kmeans.labels_
Two_clusters_labels = list(kmeans.labels_)
pd.plotting.parallel_coordinates(origin_3d_pca, 'labels', color=('#556270', '#C7F464'))
#3D plotting
origin_3d_pca.rename(index=str, columns={0:'zero', 1:'first', 2:'second'}, inplace=True)
origin_3d_pca.labels[origin_3d_pca.labels == 0] = 'negative' 
origin_3d_pca.labels[origin_3d_pca.labels == 1] = 'positive'
import plotly            as py
from plotly.offline      import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 
#NOTEBOOK WHILE KERNEL IS RUNNING


data = []
clusters = []
colors = ['rgb(228,26,28)','rgb(55,126,184)','rgb(77,175,74)']

for i in range(len(origin_3d_pca.labels.unique())):
    name = origin_3d_pca.labels.unique()[i]
    color = colors[i]
    x = origin_3d_pca[ origin_3d_pca['labels'] == name ]['zero']
    y = origin_3d_pca[ origin_3d_pca['labels'] == name ]['first']
    z = origin_3d_pca[ origin_3d_pca['labels'] == name ]['second']
    
    trace = dict(
        name = name,
        x = x, y = y, z = z,
        type = "scatter3d",    
        mode = 'markers',
        marker = dict( size=3, color=color, line=dict(width=0) ) )
    data.append( trace )
    
    cluster = dict(
        color = color,
        opacity = 0.3,
        type = "mesh3d",    
        x = x, y = y, z = z )
    data.append( cluster )

layout = dict(
    width=800,
    height=550,
    autosize=False,
    title='Market Customer Segmentation',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'        
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig)
del origin_3d_pca['labels']
origin_3d_pca.head(n=20)
kmeans = KMeans(n_clusters=5, random_state=42).fit(origin_3d_pca)
origin_3d_pca['labels'] = kmeans.labels_
Five_clusters_labels = list(kmeans.labels_)
origin_3d_pca.labels[origin_3d_pca.labels == 0] = 'a' 
origin_3d_pca.labels[origin_3d_pca.labels == 1] = 'b'
origin_3d_pca.labels[origin_3d_pca.labels == 2] = 'c' 
origin_3d_pca.labels[origin_3d_pca.labels == 3] = 'd'
origin_3d_pca.labels[origin_3d_pca.labels == 4] = 'e'
data = []
clusters = []
colors = ['rgb(228,26,28)', 'rgb(55,126,184)', 
          'rgb(77,175,74)', 'rgb(0,255,199)', 
          'rgb(0,0,255)']

for i in range(len(origin_3d_pca.labels.unique())):
    name = origin_3d_pca.labels.unique()[i]
    color = colors[i]
    x = origin_3d_pca[ origin_3d_pca['labels'] == name ]['zero']
    y = origin_3d_pca[ origin_3d_pca['labels'] == name ]['first']
    z = origin_3d_pca[ origin_3d_pca['labels'] == name ]['second']
    
    trace = dict(
        name = name,
        x = x, y = y, z = z,
        type = "scatter3d",    
        mode = 'markers',
        marker = dict( size=3, color=color, line=dict(width=0) ) )
    data.append( trace )
    
    cluster = dict(
        color = color,
        opacity = 0.3,
        type = "mesh3d",    
        x = x, y = y, z = z )
    data.append( cluster )

layout = dict(
    width=800,
    height=550,
    autosize=False,
    title='Market Customer Segmentation(Five cluster)',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        yaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        zaxis=dict(
            gridcolor='rgb(255, 255, 255)',
            zerolinecolor='rgb(255, 255, 255)',
            showbackground=True,
            backgroundcolor='rgb(230, 230,230)'
        ),
        aspectratio = dict( x=1, y=1, z=0.7 ),
        aspectmode = 'manual'        
    ),
)

fig = dict(data=data, layout=layout)
iplot(fig)

Two_clusters  = origin.copy()
Five_clusters = origin.copy()
Two_clusters['labels'] = Two_clusters_labels
Five_clusters['labels'] = Five_clusters_labels
columns=['Annual Income (k$)','Spending Score (1-100)', 'Age']
hue_col = 'labels'

Two_clusters_instance = pre_processing(Two_clusters)
 
Two_clusters_instance.plots(columns, hue_col)
columns=['Annual Income (k$)','Spending Score (1-100)', 'Age']
hue_col = 'labels'

Five_clusters_instance = pre_processing(Five_clusters)
 

Five_clusters_instance.plots(columns, hue_col)
Two_clusters['labels']  = Two_clusters_labels
Five_clusters['labels'] = Five_clusters_labels
sns.scatterplot(x="Age", y="Annual Income (k$)",
                hue="labels", 
                sizes=(1, 8), linewidth=0,
                data=Two_clusters)
sns.scatterplot(y="Spending Score (1-100)", x="Annual Income (k$)",
                hue="labels", 
                sizes=(1, 8), linewidth=0,
                data=Two_clusters)
# 'Female', 'Male'
sns.boxplot(x="Gender", y="Annual Income (k$)", hue='labels',data=Two_clusters)
Two_clusters.head(n=20)
sns.pairplot(Two_clusters.drop(['CustomerID'], axis=1), hue="labels")
sns.scatterplot(x="Age", y="Annual Income (k$)",
                hue="labels", 
                sizes=(1, 8), linewidth=0,
                palette = ['#ff0000', '#ffc300', '#00ffff', '#00ff00', '#000000'],
                data=Five_clusters)
sns.scatterplot(y="Spending Score (1-100)", x="Annual Income (k$)",
                hue="labels", 
                sizes=(1, 8), linewidth=0,
                palette = ['#ff0000', '#ffc300', '#00ffff', '#00ff00', '#000000'],
                data=Five_clusters)
# 'Female', 'Male'
sns.boxplot(x="Gender", y="Annual Income (k$)", hue='labels',data=Five_clusters)
sns.pairplot(Five_clusters.drop(['CustomerID'], axis=1), hue="labels")
""""Final Ans:Spending score indicate, score that market 
    company measured. Basically if you're young, then whatever 
    your state of bank account company measure your spending score higher. 
    However if you're old, then whatever your state of bank account company 
    measure your spending score lower. Finally, if your income is average income, 
    then whatever your age,
    company measure your spending score at middle score.
    
    """