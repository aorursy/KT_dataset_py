# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy             as np # linear algebra

import pandas            as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn           as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.cluster       import KMeans



from tqdm import tqdm



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Mall_Customers = pd.read_csv('../input/Mall_Customers.csv')
Mall_Customers['Gender'].value_counts().plot('bar')
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

            
Mall_Customers_instance = pre_processing(Mall_Customers)
columns=['Annual Income (k$)','Spending Score (1-100)', 'Age']

hue_col = 'Gender'
Mall_Customers_instance.plots(columns, hue_col)
Mall_Customers_instance.missing_percent_plot()
len(Mall_Customers), len(Mall_Customers.CustomerID.unique())
le = LabelEncoder()

Mall_Customers.Gender = le.fit_transform(Mall_Customers.Gender)
le.classes_
Mall_Customers.head()
origin = Mall_Customers.copy()
ss = StandardScaler()

Mall_Customers = pd.DataFrame(ss.fit_transform(Mall_Customers), columns=Mall_Customers.columns)
inertia_list = []

for n_clusters in range(1, 10):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(Mall_Customers)

    inertia_list.append(kmeans.inertia_)
sns.lineplot(x= [i for i in range(1, 10)], y=inertia_list,marker=True)
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

pd.tools.plotting.parallel_coordinates(new_Mall_Customers, 'labels', color=('#556270', '#C7F464', '#FF6B6B', '#000000'))
# plot with raw data

pd.tools.plotting.parallel_coordinates(origin, 'labels', color=('#556270', '#C7F464', '#FF6B6B', '#000000'))
# plot with standardized data

ss_origin = pd.DataFrame(ss.fit_transform(Mall_Customers), columns=Mall_Customers.columns)

ss_origin['labels'] = kmeans.labels_

pd.tools.plotting.parallel_coordinates(ss_origin, 'labels', color=('#556270', '#C7F464', '#FF6B6B', '#000000'))
inertia_list = []

for n_clusters in range(1, 10):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(origin)

    inertia_list.append(kmeans.inertia_)

# with raw data

sns.lineplot(x= [i for i in range(1, 10)], y=inertia_list,marker=True)
del origin['labels']
kmeans = KMeans(n_clusters=2, random_state=42).fit(origin)

origin['labels'] = kmeans.labels_
pd.tools.plotting.parallel_coordinates(origin, 'labels', color=('#556270', '#C7F464'))
del origin['labels']
pca=PCA(n_components=3)

pca.fit(origin)

pca.explained_variance_ratio_.sum()
origin_3d_pca = pd.DataFrame(pca.fit_transform(origin))
inertia_list = []

for n_clusters in range(1, 10):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(origin_3d_pca)

    inertia_list.append(kmeans.inertia_)

    

# with raw data

sns.lineplot(x= [i for i in range(1, 10)], y=inertia_list,marker=True)
kmeans = KMeans(n_clusters=2, random_state=42).fit(origin_3d_pca)

origin_3d_pca['labels'] = kmeans.labels_
Two_clusters_labels = list(kmeans.labels_)
pd.tools.plotting.parallel_coordinates(origin_3d_pca, 'labels', color=('#556270', '#C7F464'))
origin_3d_pca.rename(index=str, columns={0:'zero', 1:'first', 2:'second'}, inplace=True)

origin_3d_pca.labels[origin_3d_pca.labels == 0] = 'negative' 

origin_3d_pca.labels[origin_3d_pca.labels == 1] = 'positive'
import plotly.graph_objs as go

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



# IPython notebook

# py.iplot(fig, filename='pandas-3d-scatter-iris', validate=False)

# url = py.plot(fig, filename='pandas-3d-scatter-iris', validate=False)
del origin_3d_pca['labels']
origin_3d_pca.head()
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



# IPython notebook

# py.iplot(fig, filename='pandas-3d-scatter-iris', validate=False)

# url = py.plot(fig, filename='pandas-3d-scatter-iris', validate=False)
origin_ss = ss.fit_transform(origin)

pca=PCA(n_components=3)

pca.fit(origin_ss)

origin_ss_pca = pd.DataFrame(pca.transform(origin_ss))

pca.explained_variance_ratio_.sum()
kmeans = KMeans(n_clusters=5, random_state=42).fit(origin_ss_pca)

origin_ss_pca['labels'] = kmeans.labels_
origin_ss_pca.labels[origin_ss_pca.labels == 0] = 'a' 

origin_ss_pca.labels[origin_ss_pca.labels == 1] = 'b'

origin_ss_pca.labels[origin_ss_pca.labels == 2] = 'c' 

origin_ss_pca.labels[origin_ss_pca.labels == 3] = 'd'

origin_ss_pca.labels[origin_ss_pca.labels == 4] = 'e'
origin_ss_pca.rename(index=str, columns={0:'zero', 1:'first', 2:'second'}, inplace=True)
data = []

clusters = []

colors = ['rgb(228,26,28)', 'rgb(55,126,184)', 

          'rgb(77,175,74)', 'rgb(0,255,199)', 

          'rgb(0,0,255)']



for i in range(len(origin_ss_pca.labels.unique())):

    name = origin_ss_pca.labels.unique()[i]

    color = colors[i]

    x = origin_ss_pca[ origin_ss_pca['labels'] == name ]['zero']

    y = origin_ss_pca[ origin_ss_pca['labels'] == name ]['first']

    z = origin_ss_pca[ origin_ss_pca['labels'] == name ]['second']

    

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



# IPython notebook

# py.iplot(fig, filename='pandas-3d-scatter-iris', validate=False)

# url = py.plot(fig, filename='pandas-3d-scatter-iris', validate=False)
del origin_ss_pca['labels']

kmeans = KMeans(n_clusters=4, random_state=42).fit(origin_ss_pca)

origin_ss_pca['labels'] = kmeans.labels_



origin_ss_pca.labels[origin_ss_pca.labels == 0] = 'a' 

origin_ss_pca.labels[origin_ss_pca.labels == 1] = 'b'

origin_ss_pca.labels[origin_ss_pca.labels == 2] = 'c' 

origin_ss_pca.labels[origin_ss_pca.labels == 3] = 'd'
data = []

clusters = []

colors = ['rgb(228,26,28)', 'rgb(55,126,184)', 

          'rgb(77,175,74)', 'rgb(0,255,199)']



for i in range(len(origin_ss_pca.labels.unique())):

    name = origin_ss_pca.labels.unique()[i]

    color = colors[i]

    x = origin_ss_pca[ origin_ss_pca['labels'] == name ]['zero']

    y = origin_ss_pca[ origin_ss_pca['labels'] == name ]['first']

    z = origin_ss_pca[ origin_ss_pca['labels'] == name ]['second']

    

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

    title='Market Customer Segmentation(Four cluster)',

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



# IPython notebook

# py.iplot(fig, filename='pandas-3d-scatter-iris', validate=False)

# url = py.plot(fig, filename='pandas-3d-scatter-iris', validate=False)
del origin_ss_pca['labels']

kmeans = KMeans(n_clusters=2, random_state=42).fit(origin_ss_pca)

origin_ss_pca['labels'] = kmeans.labels_



origin_ss_pca.labels[origin_ss_pca.labels == 0] = 'a' 

origin_ss_pca.labels[origin_ss_pca.labels == 1] = 'b'
data = []

clusters = []

colors = ['rgb(228,26,28)', 'rgb(55,126,184)']



for i in range(len(origin_ss_pca.labels.unique())):

    name = origin_ss_pca.labels.unique()[i]

    color = colors[i]

    x = origin_ss_pca[ origin_ss_pca['labels'] == name ]['zero']

    y = origin_ss_pca[ origin_ss_pca['labels'] == name ]['first']

    z = origin_ss_pca[ origin_ss_pca['labels'] == name ]['second']

    

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

    title='Market Customer Segmentation(Two cluster)',

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



# IPython notebook

# py.iplot(fig, filename='pandas-3d-scatter-iris', validate=False)

# url = py.plot(fig, filename='pandas-3d-scatter-iris', validate=False)
Two_clusters  = origin.copy()

Five_clusters = origin.copy()
Two_clusters['labels'] = Two_clusters_labels

Five_clusters['labels'] = Five_clusters_labels
columns=['Annual Income (k$)','Spending Score (1-100)', 'Age']

hue_col = 'labels'



Two_clusters_instance = pre_processing(Two_clusters)

 

Two_clusters_instance.value_symmetry(hue_col)

Two_clusters_instance.plots(columns, hue_col)
columns=['Annual Income (k$)','Spending Score (1-100)', 'Age']

hue_col = 'labels'



Five_clusters_instance = pre_processing(Five_clusters)

 

Five_clusters_instance.value_symmetry(hue_col)

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
Two_clusters.head()
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