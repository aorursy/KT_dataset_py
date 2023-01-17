import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 6

from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
# Upload data
movies = pd.read_csv('../input/movies.csv', encoding = "ISO-8859-1")

# Have a look on the data structure and types
movies.head()
# Check if there are missing observations
movies.info()
for index in ['budget', 'gross', 'runtime', 'score', 'votes', 'year']:
    print(index, 'min =', movies[index].min(), '&', index, 'max =', movies[index].max())
len(movies[movies['budget'] == 0])
labelencoder_X = LabelEncoder()
movies.loc[:, 'company'] = labelencoder_X.fit_transform(movies.loc[:, 'company'])
movies.loc[:, 'country'] = labelencoder_X.fit_transform(movies.loc[:, 'country'])
movies.loc[:, 'director'] = labelencoder_X.fit_transform(movies.loc[:, 'director'])
movies.loc[:, 'genre'] = labelencoder_X.fit_transform(movies.loc[:, 'genre'])
movies.loc[:, 'name'] = labelencoder_X.fit_transform(movies.loc[:, 'name'])
movies.loc[:, 'rating'] = labelencoder_X.fit_transform(movies.loc[:, 'rating'])
movies.loc[:, 'star'] = labelencoder_X.fit_transform(movies.loc[:, 'star'])
movies.loc[:, 'writer'] = labelencoder_X.fit_transform(movies.loc[:, 'writer'])
# Break down (released date) into (year, month & day)
movies['rel_year'] = movies['released'].apply(lambda x: x[:4]).astype(int)
movies['rel_month'] = 0
movies['rel_day'] = 0

# Update 'rel_month' and 'rel_day' columns with the corresponding values from 'released' feature
for index in range(0, len(movies)):
    if len(movies.released[index]) == 10:
        movies.loc[index, 'rel_month'] = movies.loc[index, 'released'][5:7]
        movies.loc[index, 'rel_day'] = movies.loc[index, 'released'][8:10]
        
# For 'released' observations with length less than 10, we can update their corresponding 'rel_month' & 'rel_day' with median
month_avg = movies[movies['rel_month'] != 0]['rel_month'].median()
day_avg = movies[movies['rel_day'] != 0]['rel_day'].median()

# Update observations with length less than 10 with the average of each feature
for index in range(0, len(movies)):
    if len(movies.released[index]) != 10:
        movies.loc[index, 'rel_month'] = month_avg
        movies.loc[index, 'rel_day'] = day_avg

# Convert 'rel_month' & 'rel_day' types to (int)
movies['rel_month'] = movies['rel_month'].astype(int)
movies['rel_day'] = movies['rel_day'].astype(int)

# Delete 'released' feature since it has no importance now
del(movies['released'])

# Replace zero 'budget' observations with 'budget' median
budget_avg = movies['budget'].median()

for index in range(0, len(movies)):
    if movies.budget[index] == 0:
        movies.loc[index, 'budget'] = budget_avg

# Create a copied dataset to to group some of its features' values for a better visualization through different cross tables
data = movies.copy()

# Group the values of year feature
for number in range(1980, 2020, 4):
    data.loc[(data['year'] > number) & (data['year'] <= (number + 4)), 'year'] = number + 4
# Create lists that hold features according to each needed computation (total, sum or average)
total = ['company', 'country', 'director', 'genre', 'name', 'star', 'writer', 'rating']
summing = ['budget', 'gross', 'votes']
aver = ['runtime', 'score']

# Display each feature's progress over consecutive time intervals (4 years) using the relevant computation
for index in range(0, len(total)):
    print("Progress of '%s' over time:\n" % total[index])
    for year in data.year.unique():
        print('Total unique %s in the interval ending in %i is: %i' % (total[index], year, data[data['year'] == year][total[index]].nunique()))
    print('\n')

print('----------------------------------------------------------------------------\n')

for index in range(0, len(summing)):
    print("Progress of '%s' over time:\n" % summing[index])
    for year in data.year.unique():
        print('The sum of %s in the interval ending in %i is: %i million' % (summing[index], year, data[data['year'] == year][summing[index]].sum() / 1000000))
    print('\n')

print('----------------------------------------------------------------------------\n')

for index in range(0, len(aver)):
    print("Progress of '%s' over time:\n" % aver[index])
    for year in data.year.unique():
        print('%s median in the interval ending in %i is: %i' % (aver[index], year, data[data['year'] == year][aver[index]].median()))
    print('\n')
# Set the dimensions of the headmap
plt.figure(figsize=(15,12))

# Set a title and plot the heatmap
plt.title('Correlation of Features', fontsize=20)
sns.heatmap(data.corr().astype(float).corr(),vmax=1.0, annot=True)
plt.show()
# Define the function with it's parameters (below is the description of each parameter):

# data: The data that will be used to train the model in order to form the clusters
# features: Which features from the dataset will be considered in the clustering
# display_clusters: Specify two features to check how clustering between them looks like
# clusters: If 0, the model will calculate the best number of clusters to be used. If a value has been set manually, the user can decide the number of clusters s/he wants to cluster upon
# predict: If (True), the model will check which cluster each observation in the sample dataset (new_data) belongs to
# display_methods: If (True), the elbow and dendrogram graphs 
# centroids: Display the centroid of each cluster

def clusters (data = pd.DataFrame(), features = [], display_clusters = [], clusters = 0, predict = False, display_methods = False, centroids = False):
    
    # Create a list of features from the provided data's columns' in case the features are not provided as argument
    if (len(features) == 0):
        features = []
        for index in range(0, len(data.columns)):
                features.append(data.columns[index])

    # Consider only the specified features if the user placed a value for it while calling the function 
    else:
        data = data.loc[:, features]
    
    # Create 'X' object
    X = data.values
    
    # List to store the within clusters sum of squares (wcss) for each number of clusters to check at which number of clusers the value of wcss will not be decreased significantly anymore
    wcss = []
    
    # Set the range that will be used in several steps below in case of manually defining number of clusters 
    if clusters != 0:
        clust_range = range(1, clusters + 1)
    
    # Set the range that will be used in several steps below in case the model will detect the best number of clusters
    else:
        clust_range = range(1, 11)

    # Detect the value of wcss corresponding to each number of clusters within the specified range above 
    for i in clust_range:
        kmeans = KMeans(n_clusters = i, init = 'random', max_iter = 300, n_init = 10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # List that will store the differences in wcss percentages' changes in order to determine the best number of clusters in which any number of clusters after it won't drop the wcss value significantly
    difference = []
    
    # determine the best number of clusters (in case it's not set manually)
    if (clusters == 0): 
        for i in range(0, 8):
            difference.append((1 - (wcss[i + 1] / wcss[i])) - (1 - (wcss[i + 2] / wcss[i + 1])))
        clusters = difference.index(max(difference)) + 2
    
    # Create an object with the corresponding cluster for each observation
    kmeans = KMeans(n_clusters = clusters, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
    y_kmeans = kmeans.fit_predict(X)
    
    # List of colors that will differentiate each cluster
    color = ['red', 'blue', 'green', 'brown', 'blueviolet', 'black', 'lightgrey', 'olive', 'peru', 'yellow']
    
    # Display a graph that will show the clusters associated with the observations of any specified two features    
    if (len(display_clusters) == 2):
        for feature in features:
            if display_clusters[0] == feature:
                for feature in features:
                    if display_clusters[1] == feature:
                        for cluster in range(0, clusters):
                            plt.scatter(X[y_kmeans == cluster, features.index(display_clusters[0])], X[y_kmeans == cluster, features.index(display_clusters[1])], s = 50, c= color[cluster], label = str(cluster))
                        if (centroids):
                            plt.scatter(kmeans.cluster_centers_[:, features.index(display_clusters[0])], kmeans.cluster_centers_[:, features.index(display_clusters[1])], s = 100, c= 'orange', label = 'Centroids')
                        plt.title('Clustering %s & %s' % (display_clusters[0], display_clusters[1]))
                        plt.xlabel(display_clusters[0])
                        plt.ylabel(display_clusters[1])
                        plt.legend()
                        plt.show()
    
    # Display the elbow method and dendrogram graphically (if display_methods = True)
    if (display_methods):
        
        plt.plot(clust_range, wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.show()

        dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
        plt.title('Dendrogram')
        plt.ylabel('Euclidean distances')
        plt.show()

    # Predict the cluster of each observation in the sample dataset 'new_data' (if predict = True)
    if (predict):
        global new_data
        new_data = data.copy()
        new_data['cluster'] = kmeans.predict(new_data.values)
        display(new_data)
        print('\nThe above dataset is stored under the name of "new_data"')
    
clusters(data = movies, features = ['budget', 'year'], display_clusters = ['budget', 'year'], display_methods = True)

clusters(data = movies, display_clusters = ['gross', 'year'], clusters = 4, centroids = True)
clusters(data = movies, features = ['gross', 'votes', 'writer'], clusters = 5, predict = True)