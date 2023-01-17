# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings

import os

warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

filename = "/kaggle/input/modified-lol-champions-dataset/LoL-Champions.csv"

df = pd.read_csv(filename)

df.head()
df.describe()
df.dtypes
#Champion name set to index

df=df.set_index('Name')

df.head(5)
df.head(20)
#Change DamageType to a numeric variable

#1 for magical

#3 for physical



df.loc[df['DamageType'].astype(str).str.match('M'), 'DamageType'] = 1

df.loc[df['DamageType'].astype(str).str.match('PM'), 'DamageType'] = 2

df.loc[df['DamageType'].astype(str).str.match('P'), 'DamageType'] = 3

df.tail(20)
#check after damage type change.

df.dtypes
#Prepare data for training.

#Remove all non numerical data, which are not relevant anyway. Class will be joined later onwards.

df_train = df.drop(['Id','Class',], axis=1)
#perform k_means clustering

features = list(df_train.columns)

data = df_train[features]
#check for missing Values

nan_rows = data[data.isnull().T.any().T]

nan_rows.head()

#Ekko has Functionality NaN for some reason. Check with Riot site. Functionality 1.

data = data.fillna(1)
#We will make 6 clusters as Riot's initial classes were 6.

#Support, Warrior, Tank, Marksman, Assassin, Mage

clustering_kmeans = KMeans(n_clusters=6, precompute_distances="auto", n_jobs=100)
data['clusters'] = clustering_kmeans.fit_predict(data).astype(int)
print(data.info())
data.head(5)
df_corr = df.drop(['Id','Class',], axis=1)

VarCorr = df_corr.corr()

print(VarCorr)

sns.heatmap(VarCorr,xticklabels=VarCorr.columns,yticklabels=VarCorr.columns)
#Plotting

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from matplotlib import colors

import matplotlib.cm as cm
#plt.scatter(data.Style, data.Difficulty, s=300, c=data.clusters)



fig, ax = plt.subplots()

colormap = cm.viridis

colorlist = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 0.9, 8)]

LABEL_COLOR_MAP = {0 : 'red',

                   1 : 'blue',

                   2 : 'green',

                   3 : 'cyan',

                   4 : 'magenta',

                   5 : 'orange'

                   }

label_color = [LABEL_COLOR_MAP[l] for l in data['clusters']]





#for i,c in enumerate(colorlist):



 #   x = data['Style'][i]

 #   y = data['Difficulty'][i]

 #   l = data['clusters'][i]

scatter = ax.scatter(x=data['Style'], y = data['Difficulty'], s=300, linewidth=0.1, c=label_color)



plt.xlabel('Style', fontsize = 14)

plt.ylabel('Difficulty', fontsize = 14)





plt.show()
print(clustering_kmeans.cluster_centers_)

print(type(clustering_kmeans.cluster_centers_))



pd.DataFrame(clustering_kmeans.cluster_centers_).to_csv('results0409.csv', index=False)

#Cluster 1

#Modified to include Class and Damage Type

data_0 = data.reset_index().merge(df,how="left").set_index(data.index.names)

data_0 = data_0[data_0['clusters'] == 0]

data_0 = data_0.drop(['Id'],axis=1)

data_0 = data_0.drop_duplicates()

data_0.head(40)
data_1 = data.reset_index().merge(df,how="left").set_index(data.index.names)

data_1 = data_1.drop(['Id'],axis=1)

data_1 = data_1[data_1['clusters'] == 1]

data_1 = data_1.drop_duplicates()

data_1.head(40)
data_2 = data.reset_index().merge(df,how="left").set_index(data.index.names)

data_2 = data_2.drop(['Id'],axis=1)

data_2 = data_2[data_2['clusters'] == 2]

data_2 = data_2.drop_duplicates()

data_2.head(40)
data_3 = data.reset_index().merge(df,how="left").set_index(data.index.names)

data_3 = data_3.drop(['Id'],axis=1)

data_3 = data_3[data_3['clusters'] == 3]

data_3 = data_3.drop_duplicates()

data_3.head(30)
data_4 = data.reset_index().merge(df,how="left").set_index(data.index.names)

data_4 = data_4.drop(['Id'],axis=1)

data_4 = data_4[data_4['clusters'] == 4]

data_4 = data_4.drop_duplicates()

data_4.head(30)
data_5 = data.reset_index().merge(df,how="left").set_index(data.index.names)

data_5 = data_5.drop(['Id'],axis=1)

data_5 = data_5[data_5['clusters'] == 5]

data_5 = data_5.drop_duplicates()

data_5.head(30)