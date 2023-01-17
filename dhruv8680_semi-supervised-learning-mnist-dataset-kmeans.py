#Libraries used
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
def group_cluster_labels(cluster_labels, number_of_data_points, new_labels):
    '''
        Recreation/re-assigning of labels to cluster labels or 
        grouping up the different cluster by there visual appearance
        
        IMPORTANT PART FOR SEMI-SUPERVISED LEARNING
        
        @cluster_labels: kmeans.labels_ | labels generated from clustering algorithms
        @number_of_data_points: length of your train data to create empty array of same size
        @new_labels: new labels to group the cluster labels. new_labels and cluster_labels must be of same size
        
        return: updated labels
    '''
    updated_labels = np.empty(number_of_data_points, dtype=np.int32)
    for i in range(k):
        updated_labels[kmeans.labels_==i] = new_labels[i]
    return updated_labels

def display_cluster_centers(cluster_center_vector, size=[8,8]):
    '''
        Display all the cluster center images to find the groups.
        
        @cluster_center_vector: flatten cluster center image array to display. 
    '''
    r,c = np.ceil(np.sqrt(len(cluster_center_vector))), np.ceil(np.sqrt(len(cluster_center_vector))) 
    plt.figure(figsize=(7,7))
    for idx, dig in enumerate(cluster_center_vector):
        plt.subplot(r, c, idx+1)
        plt.imshow(dig.reshape(size), cmap="gray")
        plt.axis('off')
        
def automatic_label_grouping(pair_dist, n_clusters, threshold=30):
    '''
        Automatic grouping of similar clusters based on the centers. 
        
        @pair_dist: pair distances of cluster center data points
        @n_clusters: number of clusters
        @threshold: threshold is minimum value of distance to be used to group clusters
        
        return: labels of grouped clusters
    '''
    temp_labels = [i for i in range(n_clusters)]
    for i in range(len(pair_dist)):
        for j in range(i+1, len(pair_dist)):
            if pair_dist[i, j] < threshold:
                temp_labels[j] = temp_labels[i]
    return temp_labels

def get_closest_data_points_to_cluster_centers(cluster_distance, cluster_labels, x_train, y_train, closest_percent=70):
    '''
        This is used to find the data points closest to the cluster center by setting up the threshold as percentage.
        Can to used to find the outliers
        
        @cluster_distance: distances from cluster centers generated from clustering (cluster_distance = kmeans.fit_transform)
        @cluster_labels: labels generated from clustering
        @x_train: dependent variable or x training data points matrix
        @y_train: Can be cluster labels or manually grouped labels or true labels
        @closest_percent: percentage to get the data point to cluster center
        
        return: top n percent x_train, y_train, group_indexes(index values of data points)
    '''
    updated_cluster_distance = cluster_distance[np.arange(len(x_train)), cluster_labels]
    for i in range(k):
        _cluster_labels = (cluster_labels == i)
        _cluster_dist = updated_cluster_distance[_cluster_labels]
        cutoff_distance = np.percentile(_cluster_dist, closest_percent)
        above_cutoff = (updated_cluster_distance > cutoff_distance)
        updated_cluster_distance[_cluster_labels & above_cutoff] = -1

    group_indexes = (updated_cluster_distance != -1)
    group_indexes = [key for key, val in enumerate(group_indexes) if val]
    return x_train[group_indexes], y_train[group_indexes], group_indexes

def display_paired_distances(cluster_center_data_points):
    '''
        Display the heatmap of paired distances
        
        @cluster_center_data_points: feature vectors
        return: paired distances
    '''
    pair_dist = pairwise_distances(cluster_center_data_points)
    plt.figure(figsize=(15,10))
    _ = sns.heatmap(pair_dist, cmap="gray")
    return pair_dist

X, y = load_digits(return_X_y=True)
k = 50
kmeans = KMeans(n_clusters=k)
X_cluster_dist = kmeans.fit_transform(X)
center_digit_idx = np.argmin(X_cluster_dist, axis=0)
X_center_digits = X[center_digit_idx]
display_cluster_centers(X_center_digits)
manual_y_group_digits = [8,9,2,6,7,5,0,4
                       ,7,1,3,2,5,2,8,8
                       ,4,6,1,0,0,7,1,9
                       ,3,5,4,3,6,5,1,3
                       ,4,6,4,7,9,2,2,8
                       ,3,9,3,1,7,1,0,7
                       ,4,1]
manual_y_grouped_labels = group_cluster_labels(kmeans.labels_, len(X), manual_y_group_digits)
manual_train_x, manual_train_y, manual_train_indexes = get_closest_data_points_to_cluster_centers(X_cluster_dist,
                                                                                                   kmeans.labels_,
                                                                                                   X,
                                                                                                   manual_y_grouped_labels,
                                                                                                   closest_percent=70)
manual_images_x=[]
for v in range(0,10):
    manual_images_x+=list(manual_train_x[manual_train_y==v][:10])
inc = 1
plt.figure(figsize=(10,9))
for inc, img in enumerate(manual_images_x):
    plt.subplot(10,10,inc+1)
    plt.imshow(img.reshape([8,8]), cmap="gray")
    plt.axis("off")
    inc+=1
#     plt.show()
dist_pair = display_paired_distances(X_center_digits)
auto_labels = automatic_label_grouping(dist_pair, k, threshold=30)
auto_train_labels = group_cluster_labels(kmeans.labels_, len(X), auto_labels)
auto_train_x, auto_train_y, auto_train_indexes = get_closest_data_points_to_cluster_centers(X_cluster_dist,
                                                                                           kmeans.labels_,
                                                                                           X,
                                                                                           auto_train_labels,
                                                                                           closest_percent=70)
auto_images_x=[]
for v in list(set(auto_labels)):
    auto_images_x+=list(auto_train_x[auto_train_y==v][:10])
inc = 1
plt.figure(figsize=(10,len(auto_labels)))
for inc, img in enumerate(auto_images_x):
    plt.subplot(len(auto_labels),10,inc+1)
    plt.imshow(img.reshape([8,8]), cmap="gray")
    plt.axis("off")
    inc+=1
#     plt.show()