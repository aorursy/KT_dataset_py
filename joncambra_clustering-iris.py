import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering, KMeans
data = pd.read_csv('../input/iris.csv')
data.head(10) # show the first 10 lines
data.describe()
categorical_summaries = [data[c].value_counts() for c in data.columns if data[c].dtype == 'object']

for i in categorical_summaries:
    display(pd.DataFrame(i))
def numerical_distribution (colnumber, plot_type='histogram', data=data):
    """
    function for plotting histogram of the column number corresponding to the numerical variable selected
    colnumber is the column number corresponding to the numerical variable selected,
    a boxplot of the numerical variable depending on the categorical variable
    """
    
    if plot_type=='histogram':
        
        # x label for the histogram
        plt.xlabel(data.columns[colnumber])
    
        # y label for the histogram (-1 points to the last column of the set)
        plt.ylabel('Frequency')
    
        # title for the histogram
        plt.title(data.columns[colnumber] + ' distribution')
    
        # histogram
        data.iloc[:,colnumber].plot.hist()
        
    elif plot_type=='boxplot':
    
        # setting type of plot
        sns.set(style="ticks", color_codes=True)
    
        # setting what values we plot
        sns.catplot(x=data.columns[-1], y=data.columns[colnumber], kind='box',data=data);
    
        # title
        plt.title(data.columns[colnumber] + ' distribution depending on ' + data.columns[-1])
numerical_distribution(0,'histogram')
numerical_distribution(0,'boxplot')
numerical_distribution(1,'histogram')
numerical_distribution(1,'boxplot')
numerical_distribution(2,'histogram')
numerical_distribution(2,'boxplot')
numerical_distribution(3,'histogram')
numerical_distribution(3,'boxplot')
g = sns.PairGrid(data, hue="species")
g.map_diag(plt.hist, alpha=0.5)
g.map_upper(plt.scatter, alpha=0.5, marker='x')
g.map_lower(sns.kdeplot, shade=True, shade_lowest=False, alpha=0.4)
g.add_legend()
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
# splitting compulsory
# X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.2)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # within cluster sum of squares
plt.show()
# creating the k-means classifier with n_clusters being k, the number of clusters for the model
kmeans = KMeans(n_clusters = 3 , 
                init = 'k-means++', 
                max_iter = 300, 
                n_init = 10, 
                random_state = 0)
# fitting the model
y_kmeans = kmeans.fit_predict(X)
def plot_data_cluster_output(method, y_clusters, features,
                             output, k_clusters, x_axis, y_axis):
    """
    function plotting the data labeled by clusters and by output
    """
    # plotting the points labeled by the cluster
    # legend
    fig, (ax1,ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 8)
    colors = iter(cm.rainbow(np.linspace(0, 1, k_clusters)))
    
    for i in range(0,k_clusters):
        ax1.scatter(features.iloc[y_clusters == i, x_axis], features.iloc[y_clusters == i, y_axis], s = 30, color= next(colors), label = 'cluster ' + str(i))
    ax1.legend()
    
    # plotting the centroid
    if method=='kmeans':
        # plotting the centroids of the clusters
        ax1.scatter(kmeans.cluster_centers_[:, x_axis], kmeans.cluster_centers_[:,y_axis], s = 30, c = 'yellow', label = 'centroids')
    
    # title
    ax1.title.set_text(method + " clustering on " + features.columns[x_axis] + " and " + features.columns[y_axis])
    # x label
    ax1.set_xlabel(features.columns[x_axis])
    # y label
    ax1.set_ylabel(features.columns[y_axis])
    
    # preparing legend to get the same colour if the number of clusters is equal to the number of labels
    if len(output.unique())==k_clusters:
        # initialize labels vector
        labels=[None]*3
        
        # reordering labels
        for i in range(0,k_clusters):
            index_label=np.where(np.amax(pd.crosstab(y_kmeans, output).iloc[:,i].values)==pd.crosstab(y_kmeans, output).iloc[:,i].values)[0]
            labels[index_label[0]]=output.unique()[i]
    else:
        labels=output.unique()
    
    plt.figure(2)
    colors = iter(cm.rainbow(np.linspace(0, 1, len(output.unique()))))
    
    # plotting the points labeled by the label
    for i in labels:
        ax2.scatter(features.iloc[output.values == i, x_axis], features.iloc[output.values == i, y_axis], s = 30, color = next(colors), label = i)   
    ax2.legend()
    # title
    ax2.title.set_text("Data labeled on species depending on " + features.columns[x_axis] + " and " + features.columns[y_axis])
    # x label
    ax2.set_xlabel(features.columns[x_axis])
    # y label
    ax2.set_ylabel(features.columns[y_axis])
plot_data_cluster_output('kmeans', y_kmeans, X, y, 3, 0, 1)
plot_data_cluster_output('kmeans', y_kmeans, X, y, 3, 0, 2)
pd.crosstab(y_kmeans, y)
print("Percentage of non-corresponding clusters:" + str(16*100/150) +"%")
def silhouette(y_clusters, k_clusters, features=X):
    """
    Function plotting the clustering configuration silhouette and average value
    arguments: y_cluster= clustering output
               k_clusters= number of clusters
               features=features used for clustering
    """
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)

    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-1, 1])

    # The (k_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to separate them clearly.
    ax1.set_ylim([0, len(features) + (k_clusters + 1) * 10])

    # silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(features, y_clusters)
    print("For n_clusters =", k_clusters ,
          "the average silhouette_score is:", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(features, y_clusters)

    y_lower = 10
    for i in range(k_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y_clusters == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next cluster plot
        y_lower = y_upper + 10 

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # no yaxis labels / ticks
    ax1.set_xticks([-1,-0.5,-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for clustering on data "
                      "with n_clusters = %d" % k_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()
silhouette(y_kmeans,3,X)
# creating the k-means classifier with n_clusters being k, the number of clusters for the model
k_5_means = KMeans(n_clusters = 5 , 
                init = 'k-means++', 
                max_iter = 300, 
                n_init = 10, 
                random_state = 0)
# fitting the model
y_5_kmeans = k_5_means.fit_predict(X)
silhouette(y_5_kmeans,5,X)
# generate the linkage matrix
Z = linkage(X, 'ward')

# we set the cut off, the level where we decide to cluster the data, we select the distance level where the line does not intercept any node.
# try different levels to understand this concept
max_d = 8               

plt.figure(figsize=(20, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
dendrogram(
    Z,
    truncate_mode='lastp', 
    p=150,                  # indicates how many final leafs you want to have ( ideally the number of samples)
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=8.,      # font size for the x axis labels
)
plt.axhline(y=max_d, c='k')
plt.show()
# Creating the hierarchical clustering model with n_clusters declaring the number of clusters
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')  
y_hier=cluster.fit_predict(X)
silhouette(y_hier,3,X)
plot_data_cluster_output('HCA',y_hier,X, y, 3, 0, 1)
plot_data_cluster_output('HCA',y_hier,X, y, 3, 0, 2)
pd.crosstab(y_hier, y)
print("percentage of non-corresponding clusters:" + str(16*100/150) +"%")