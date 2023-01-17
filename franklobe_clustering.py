import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
class Cluster_Sum_Squares():
    
    file_features = ''
    file_name = ''
    
    def plot_wcss(self, wcss):
        
        number_clusters = range(1,len(self.file_features))
        plt.plot(number_clusters, wcss)
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
    
    def get_wcss(self):
        
        wcss = []
        
        for count in range(1,len(self.file_features)):
            
            kmeans = KMeans(count)
            kmeans.fit(self.file_features)
            wcss_inertia = kmeans.inertia_
            wcss.append(wcss_inertia)
        
        self.plot_wcss(wcss)              
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_features = file_data.iloc[:,0:2]
    
    def __init__(self):
        
        pass
    
def within_cluster_sum_squares():
    
    cluster_sum_squares = Cluster_Sum_Squares()
    cluster_sum_squares.file_name = '../input/satisfaction-loyalty/Satisfaction-Loyalty.csv'
    cluster_sum_squares.load_file()
    cluster_sum_squares.get_wcss()
    
within_cluster_sum_squares()
class Clustering():
    
    file_features = ''
    file_name = ''
    
    def get_result(self):
        
        kmeans = KMeans(9)
        kmeans.fit(self.file_features)
        file_clusters = kmeans.fit_predict(self.file_features)
        self.file_features['Cluster'] = file_clusters
        
        self.plot_result()
    
    def load_file(self):
        
        file_data = pd.read_csv(self.file_name)
        self.file_features = file_data.iloc[:,0:2]
        
    def plot_result(self):
        
        print(self.file_features)
        
        plt.scatter(self.file_features['Satisfaction'], self.file_features['Loyalty'], c = self.file_features['Cluster'], cmap = 'rainbow')
        plt.xlabel('Satisfaction', fontsize = 20)
        plt.ylabel('Loyalty', fontsize = 20)
    
    def __init__(self):
        
        pass
    
def perform_clustering():
    
    clustering = Clustering()
    clustering.file_name = '../input/satisfaction-loyalty/Satisfaction-Loyalty.csv'
    clustering.load_file()
    clustering.get_result()
    
perform_clustering()
