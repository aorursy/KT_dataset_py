import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn import decomposition

import plotly.express as px
# Reading CSV file

data = pd.read_csv('../input/pubg-statisctic/PUBG.csv')

data.head()
# We will use only solo player data one can use squad or duo its up to him/her. So removing unnecessary columns



data.dropna(inplace=True)



columns_to_be_removed = np.arange(52,152,1)



data.drop(data.columns[columns_to_be_removed],axis=1,inplace = True)   # Removing columns form 52 to 151



data.drop(data.columns[[0,1]], axis = 1, inplace = True)               # Removing player_name and tracker_id



data.drop(columns= ['solo_Revives'],inplace = True)                    # There is no teammate to revive that is why removing



data.drop(columns= ['solo_DBNOs'], inplace = True)                     # DBNOs = Knock Outs, in solo game no knock out so always = 0





# By using data.shape we can see that there is 48 features left for solo player, uncomment below code



#data.shape

# Its just one line of code...



standardized_data = StandardScaler().fit_transform(data)

standardized_data.shape
pca = decomposition.PCA()                         # Just simple varible assignment



pca.n_components = 48                             # Total numbers of features



pca_data = pca.fit_transform(standardized_data)   # Fitting all the features to the PCA



info_explained_by_each_feature = pca.explained_variance_ / np.sum(pca.explained_variance_)      # Calculating information explained by each feature



cum_info_explained = np.cumsum(info_explained_by_each_feature)    # Cumulative sum of information explained by each feature





# Plotting Graph



plt.figure(figsize = (10,5))

plt.plot(cum_info_explained)

plt.title('Numbers of features vs Information Explained')

plt.xlabel('Numbers of features')

plt.ylabel('Cumulative sum of % Information explained');
pca = decomposition.PCA()



pca.n_components = 20                              # we want 20 most informative features.



pca_data = pca.fit_transform(standardized_data)    # It will calculate and provide features as an array



df = pd.DataFrame(pca_data)                       # Converting array in to dataframe



df
train, test = train_test_split(df,test_size = 0.2,random_state = 1)

dev , test = train_test_split(test,test_size = 0.2,random_state = 1)
ks = range(1,10)

inertias = []



for i in ks:

    model = KMeans(n_clusters = i,init = 'k-means++',random_state=1 )

    model.fit(train)

    

    inertias.append(model.inertia_)

    

    print(f'Inertia for {i} Culsters is {model.inertia_:.0f}')

    

for i in range(1,9):

    print(f'The difference between inertia of {i+1} and {i} cluster is {inertias[i-1] - inertias[i]}')
plt.figure(figsize=(12,5))

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.title('Numbers of clusters vs Inertia')

plt.plot(ks,inertias);



# We can see that after number of cluster = 4 the inertia reduces very slowly so that optimal number of clusters is 4 
kmeans = KMeans(n_clusters=4, init = 'k-means++',random_state=1).fit(train)



labels = kmeans.labels_



df_trained = pd.DataFrame(train)

df_trained['Cluster'] = pd.Series(labels)



cluster_names = {0:'Beginner',1:'Hacker',2:'Experienced',3:'Professional'}

df_trained['Cluster_names'] = df_trained['Cluster'].map(cluster_names)

df_trained.dropna(inplace = True)
# Note that the clustering is based on the 20 features means 20 Dimensions which we can not visualize so this 3d scatter plot is not looking good.



scatter = px.scatter_3d(x=0,y=1,z=2,data_frame=df_trained,color='Cluster_names')

scatter.show()