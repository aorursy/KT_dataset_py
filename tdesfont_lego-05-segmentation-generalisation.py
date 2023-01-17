from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage
import skimage.io
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float
# navigate through the existing structure
! ls ../input/lego/06-Partial_Assembly_Dataset/
path = '../input/lego/'
os.listdir(path)
path_assembly = '../input/lego/06-Partial_Assembly_Dataset/'
folder_names = sorted([name+'/' for name in os.listdir(path_assembly) if 'Partial' in name])
n_steps = 5
selected_steps = folder_names[:n_steps]
selected_steps
def compute_distance(cluster, index_cluster_1, index_cluster_2):
    min_dist = float('inf')
    for point1 in cluster[index_cluster_1]:
        for point2 in cluster[index_cluster_2]:
            distance = np.sqrt(np.sum((point1 - point2)**2))
            if min_dist > distance:
                min_dist = distance
    return min_dist
class UnionFind:
    
    def __init__(self, n):
        self.up = list(range(n))
        self.rank = [0]*n
        
    def find(self, x):
        if self.up[x] == x:
            return x
        else:
            self.up[x] = self.find(self.up[x])
            return self.up[x]
        
    def union(self, x, y):
        repr_x = self.find(x)
        repr_y = self.find(y)
        if repr_x == repr_y:
            return False
        if self.rank[repr_x] == self.rank[repr_y]:
            self.rank[repr_x] += 1
            self.up[repr_y] = repr_x
        elif self.rank[repr_x] > self.rank[repr_y]:
            self.up[repr_y] = repr_x
        else:
            self.up[repr_x] = repr_y
        return True
from collections import defaultdict
image_path = path_assembly + selected_steps[4]
image = np.random.choice(os.listdir(image_path))
img = plt.imread(image_path + image)

def pipeline_segmentation(img, treshold_spatial_distance = 5, treshold_color_distance = 0.40):
    
    # Resize image
    resized_assembly_image = resize(img, (400, 300))
    # Segment image
    segments_fz = felzenszwalb(resized_assembly_image, scale=300, sigma=0.05, min_size=100)

    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.imshow(resized_assembly_image)
    plt.subplot(122)
    plt.imshow(mark_boundaries(resized_assembly_image, segments_fz))
    plt.show()
    
    # Plot separated segmented parts
    # print('Number of segments:', len(np.unique(segments_fz)[1:]))
        
    # Connected components recombination
    n_clusters = len(np.unique(segments_fz))-1
    
    cluster_geometric_sampler = {}
    # store the points
    for cluster_id in np.unique(segments_fz)[1:]:
        segmented_image = resized_assembly_image.copy()
        mask = segments_fz == cluster_id
        xy = np.where(mask)
        points = np.array([(x, y) for (x, y) in zip(xy[0], xy[1])])
        N = len(points)
        n_samples = int(N/20)
        cluster_geometric_sampler[cluster_id] = points[np.random.choice(list(range(N)), n_samples)]
    
    values = []
    graph_spatial_distance = np.zeros((n_clusters, n_clusters))
    for index_cluster_1 in list(range(0, n_clusters)):
        for index_cluster_2 in list(range(index_cluster_1, n_clusters)):
            distance = round(compute_distance(cluster_geometric_sampler, index_cluster_1+1, index_cluster_2+1), 1)        
            graph_spatial_distance[index_cluster_1][index_cluster_2] = distance   
            if distance!=0: values.append(distance)
                
    plt.figure(figsize=(6, 3))
    plt.title('Distribution of inter-distance')
    plt.hist(values, edgecolor='k', bins=10)
    plt.show()

    cluster_color_centroids = {}
    for cluster_id in np.unique(segments_fz)[1:]:
        segmented_image = resized_assembly_image.copy()
        mask = segments_fz == cluster_id
        centroid_color = np.mean(segmented_image[mask],0)
        cluster_color_centroids[cluster_id] = centroid_color
    
    values = []
    graph_color_distance = np.zeros((n_clusters, n_clusters))
    for index_cluster_1 in list(range(0, n_clusters)):
        for index_cluster_2 in list(range(index_cluster_1+1, n_clusters)):
            centroid_1 = cluster_color_centroids[index_cluster_1+1]
            centroid_2 = cluster_color_centroids[index_cluster_2+1]
            distance = np.sqrt(np.sum((centroid_1 - centroid_2)**2))
            graph_color_distance[index_cluster_1][index_cluster_2] = distance
            values.append(distance)

    plt.figure(figsize=(6, 3))
    plt.title('Distribution of centroid color distance')
    plt.hist(values, bins=20, edgecolor='k')
    plt.grid()
    plt.show()
      
    # Connected components
    rule_spatial_union = graph_spatial_distance < treshold_spatial_distance
    rule_color_union = graph_color_distance < treshold_color_distance
    # Boolean AND 
    union = rule_spatial_union * rule_color_union
    graph = union.T * union
    
    uf = UnionFind(n_clusters+1)
    
    for i in range(0, n_clusters):
        for j in list(np.where(union[i][i+1:])[0]+i):
            uf.union(i+1, j+2)
    
    components = defaultdict(list)
    for i in range(n_clusters):
        components[uf.find(i)].append(i)
        
    for i, connect_components in enumerate(components.values()):
        if i != 0 and connect_components:
            segmented_image = resized_assembly_image.copy()
            mask = np.ones((segmented_image.shape[0], segmented_image.shape[1]), dtype='bool')
            for cluster_id in connect_components:
                mask *= segments_fz != cluster_id
            segmented_image[mask] = [1, 1, 1]
            plt.imshow(segmented_image)
            plt.show()
    
    print('Initial number of segments: {} -> Number of connected components afterwards: {}'.format(n_clusters, len(components.values())-1))

pipeline_segmentation(img, treshold_spatial_distance=5, treshold_color_distance=0.4)
image_path = path_assembly + selected_steps[0]
image = np.random.choice(os.listdir(image_path))
img = plt.imread(image_path + image)
pipeline_segmentation(img, treshold_spatial_distance=5, treshold_color_distance=0.4)
image_path = path_assembly + selected_steps[1]
image = np.random.choice(os.listdir(image_path))
img = plt.imread(image_path + image)
pipeline_segmentation(img, treshold_spatial_distance=10, treshold_color_distance=0.1)
image_path = path_assembly + selected_steps[2]
image = np.random.choice(os.listdir(image_path))
img = plt.imread(image_path + image)
pipeline_segmentation(img, treshold_spatial_distance=5, treshold_color_distance=0.4)
image_path = path_assembly + selected_steps[3]
image = np.random.choice(os.listdir(image_path))
img = plt.imread(image_path + image)
pipeline_segmentation(img, treshold_spatial_distance=5, treshold_color_distance=0.4)
image_path = path_assembly + selected_steps[4]
image = np.random.choice(os.listdir(image_path))
img = plt.imread(image_path + image)
pipeline_segmentation(img, treshold_spatial_distance=5, treshold_color_distance=0.4)