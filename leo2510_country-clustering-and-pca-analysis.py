# Load necessary library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn import metrics



import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn')

%matplotlib inline



# set default plot size

plt.rcParams["figure.figsize"] = (15,8)
# Load and preview data 

country = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')



country.head()

# print(country.shape)
# Summary Statistics

country.describe()
# Check each column for nas

country.isnull().sum()
sns.pairplot(country.drop('country',axis=1))
country_cor = country.drop('country',axis=1).corr()

country_cor
# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(country_cor, dtype=np.bool))



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(15, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(country_cor, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True)
# scale the data

min_max_scaler = MinMaxScaler()

country_scale = min_max_scaler.fit_transform(country.drop('country',axis=1))

country_scale_df = pd.DataFrame(data = country_scale,

                               columns=country.columns[1:])

country_scale_df['country'] = country['country']

country_scale_df.head()
# pass through the scaled data set into our PCA class object

pca = PCA().fit(country_scale)



# plot the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))



# define the labels & title

plt.xlabel('Number of Components', fontsize = 15)

plt.ylabel('Variance (%)', fontsize = 15) 

plt.title('Explained Variance', fontsize = 20)



# show the plot

plt.show()
# we will choose 5 pca components and create a new dataset



country_pca = PCA(n_components=5).fit(country_scale).transform(country_scale)



# store it in a new data frame

country_pca= pd.DataFrame(data = country_pca, columns = ['principal component 1', 'principal component 2',

                                                        'principal component 3','principal component 4',

                                                        'principal component 5'])

# country_pca['country'] = country['country']



country_pca.head()
country_pca_cor = country_pca.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(country_pca_cor, dtype=np.bool))



# Set up the matplotlib figure

fig, ax = plt.subplots(figsize=(15, 8))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(country_pca_cor, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot = True)
# define a dictionary that contains all of our relevant info.

results = []



# define how many clusters we want to test up to.

num_of_clusters = 10



# run through each instance of K

for k in range(2, num_of_clusters):

    

    print("-"*100)

    

    # create an instance of the model, and fit the training data to it.

    kmeans = KMeans(n_clusters=k, random_state=0).fit(country_pca)

    

    

    # store the different metrics

#     results_dict_pca[k]['silhouette_score'] = sil_score

#     results_dict_pca[k]['inertia'] = kmeans.inertia_

#     results_dict_pca[k]['score'] = kmeans.score

#     results_dict_pca[k]['model'] = kmeans

    

    results.append(kmeans.inertia_)

    

    # print the results    

    print("Number of Clusters: {}".format(k),kmeans.inertia_)

plt.plot(range(2, num_of_clusters), results, 'bx-')

plt.xlabel('k')

plt.ylabel('Sum_of_squared_distances')

plt.title('Elbow Method For Optimal k')

plt.show()
# From the graph above, it indicates that we should choose k = 3



from yellowbrick.cluster import SilhouetteVisualizer



clusters = [2,3,4]



for cluster in clusters:

    

    print('-'*100)



    # define the model for K

    kmeans = KMeans(n_clusters = cluster, random_state=0)



    # pass the model through the visualizer

    visualizer = SilhouetteVisualizer(kmeans)



    # fit the data

    visualizer.fit(country_pca)



    # show the chart

    visualizer.poof()



    

# the silhouette plot also shows that the optimal k is 3
kmeans = KMeans(n_clusters=3, random_state=0).fit(country_pca)

country['cluster'] = kmeans.labels_

country.head()
country[country['cluster'] == 0][:10]
country[country['cluster'] == 1][:10]
country[country['cluster'] == 2][:10]