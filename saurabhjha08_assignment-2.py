import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
app_data = pd.read_csv('/kaggle/input/loan-defaulter/application_data.csv')
print(list(app_data.columns))
app_data.head(5)
input_data = app_data[['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                        'OWN_CAR_AGE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'REGION_POPULATION_RELATIVE', 
                        'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 
                        'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
                        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 
                        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']].fillna(0).head(10000)
input_data
class SoftLeaderClustering:
    def __init__(self,data: pd.DataFrame, threshold: float):
        self.__data = data
        self.__threshold = threshold
        self.__data_matrix = data.to_numpy()
        self.__cluster_repr_matrix = None
        self.__assignment_matrix = None
        self.__product_matrix = None
        self.__error = None
    
    def shuffle_data(self):
        random.shuffle(self.__data_matrix)
        
    def do_clustering(self):
        self.__build_cluster_repr_matrix()
        self.__build_assignment_matrix()
        self.__build_product_matrix()
        self.__calculate_error()
    
    def get_error(self) -> float:
        return self.__error
    
    def get_number_of_clusters(self) -> int:
        return len(self.__cluster_repr_matrix)
    
    def __build_cluster_repr_matrix(self):
        cluster_leaders = []
        for data_point in self.__data_matrix:
            if len(cluster_leaders) == 0:
                cluster_leaders.append(data_point)
                continue
            else:
                is_new_leader = True
                for leader in cluster_leaders:
                    distance_from_leader = distance.euclidean(leader, data_point)
                    if distance_from_leader < self.__threshold: 
                        is_new_leader = False
                        break
                if is_new_leader: 
                    cluster_leaders.append(data_point)
        self.__cluster_repr_matrix = cluster_leaders
        
    def __build_assignment_matrix(self):
        assignment_matrix = []
        for data_point in self.__data_matrix:
            distance_arr = [np.exp(-distance.euclidean(data_point, p)) for p in self.__cluster_repr_matrix]
            sum_val = np.sum(list(filter(lambda d: d < self.__threshold, distance_arr)))
            if(sum_val != 0):
                assignment_matrix.append(np.array([np.true_divide(p, sum_val)*(1 if p < self.__threshold else 0) for p in distance_arr]))
            else:
                assignment_matrix.append(np.array(list(map(lambda v: 1 if np.array_equal(v,data_point) else 0, distance_arr))))
        self.__assignment_matrix = assignment_matrix
    
    def __build_product_matrix(self):
        self.__product_matrix = np.dot(self.__assignment_matrix, self.__cluster_repr_matrix)
    
    def __calculate_error(self):
        row_length = len(self.__data_matrix)
        col_length = len(self.__data_matrix[0])
        error_val = 0
        for row in range(row_length):
            for col in range(col_length):
                error_val += np.square(self.__data_matrix[row][col] - self.__product_matrix[row][col])
        self.__error = error_val
        
        
    
threshold_vals = [100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000]
cluster_count = []
error_val_for_thresholds = []
for threshold in threshold_vals:
    soft_cluster = SoftLeaderClustering(input_data, threshold)
    soft_cluster.do_clustering()
    cluster_count.append(soft_cluster.get_number_of_clusters())
    error_val_for_thresholds.append(soft_cluster.get_error())
cluster_count_df = pd.DataFrame({'threshold_val': threshold_vals, 'cluster_count': cluster_count, 'error': error_val_for_thresholds})
cluster_count_df
cluster_count_df.plot(x='threshold_val', y = 'cluster_count', kind="line", figsize=(5,4))
cluster_count_df.plot(x='threshold_val', y = 'error', kind="line", figsize=(5,4))
cluster_count_df.plot(x='cluster_count', y = 'error', kind="line", figsize=(5,4))
soft_cluster = SoftLeaderClustering(input_data, 200000)
num_clusters = []
error_vals = []
for shuffle_num in range(10):
    soft_cluster.shuffle_data()
    soft_cluster.do_clustering()
    num_clusters.append(soft_cluster.get_number_of_clusters())
    error_vals.append(soft_cluster.get_error())
soft_leader_clustering_results = pd.DataFrame({'shuffle_num': list(range(1,11)), 'num_clusters': num_clusters, 
                                               'error_vals': error_vals})
soft_leader_clustering_results
soft_leader_clustering_results.plot(x='num_clusters', y = 'error_vals', kind="line", figsize=(5,4))
class KMeansClustering:
    def __init__(self, data: pd.DataFrame, k: int):
        self.__data = data
        self.__k_value = k
        self.__data_matrix = data.to_numpy()
        self.__cluster_repr_matrix = None
        self.__assignment_matrix = None
        self.__product_matrix = None
        self.__error = None
    
    def do_clustering(self):
        kmeans = KMeans(n_clusters=self.__k_value, init='k-means++', random_state=0).fit(self.__data_matrix)
        self.__cluster_repr_matrix = kmeans.cluster_centers_
        self.__build_assignment_matrix(kmeans.labels_)
        self.__build_product_matrix()
        self.__calculate_error()
    
    def get_error(self) -> float:
        return self.__error
        
    
    def __build_assignment_matrix(self, labels):
        assignment_matrix = []
        for label in labels:
            label_arr = [0]*self.__k_value
            label_arr[label] = 1
            assignment_matrix.append(label_arr)
        self.__assignment_matrix = assignment_matrix
        
        
    def __build_product_matrix(self):
        self.__product_matrix = np.dot(self.__assignment_matrix, self.__cluster_repr_matrix)
    
    def __calculate_error(self):
        row_length = len(self.__data_matrix)
        col_length = len(self.__data_matrix[0])
        error_val = 0
        for row in range(row_length):
            for col in range(col_length):
                error_val += np.square(self.__data_matrix[row][col] - self.__product_matrix[row][col])
        self.__error = error_val
k_values = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
kmeans_error_vals = []
for k in k_values:
    kmeans_cluster = KMeansClustering(input_data, k)
    kmeans_cluster.do_clustering()
    kmeans_error_vals.append(kmeans_cluster.get_error())
kmeans_clustering_results = pd.DataFrame({'k_value': k_values, 'error': kmeans_error_vals})
kmeans_clustering_results
kmeans_clustering_results.plot(x='k_value', y = 'error', kind="line", figsize=(5,4))
