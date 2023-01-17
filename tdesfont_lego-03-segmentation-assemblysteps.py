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
# navigate through the existing structure
! ls ../input/lego/06-Partial_Assembly_Dataset/
path = '../input/lego/'
os.listdir(path)
path_assembly = '../input/lego/06-Partial_Assembly_Dataset/'
folder_names = sorted([name+'/' for name in os.listdir(path_assembly) if 'Partial' in name])
n_steps = 5
selected_steps = folder_names[:n_steps]
selected_steps
image_path = path_assembly + selected_steps[4]
image = np.random.choice(os.listdir(image_path))
image = 'pa_04_X6Ph3Mh2fz4WQif0O081.jpg'
plt.figure(figsize=(5, 5))
plt.title('Image to segment')
img = plt.imread(image_path + image)
plt.imshow(img)
import skimage
import skimage.io
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float
resized_assembly_image = resize(img, (400, 300))
segments_fz = felzenszwalb(resized_assembly_image, scale=300, sigma=0.05, min_size=100)
print("Felzenszwalb number of segments: {}".format(len(np.unique(segments_fz))))

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.imshow(resized_assembly_image)
plt.subplot(122)
plt.imshow(mark_boundaries(resized_assembly_image, segments_fz))
plt.show()
for cluster_id in np.unique(segments_fz)[1:]:
    segmented_image = resized_assembly_image.copy()
    mask = segments_fz != cluster_id
    segmented_image[mask] = [1, 1, 1]
    plt.imshow(segmented_image)
    plt.show()
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
def compute_distance(cluster, index_cluster_1, index_cluster_2):
    min_dist = float('inf')
    for point1 in cluster[index_cluster_1]:
        for point2 in cluster[index_cluster_2]:
            distance = np.sqrt(np.sum((point1 - point2)**2))
            if min_dist > distance:
                min_dist = distance
    return min_dist
values = []
graph_spatial_distance = np.zeros((n_clusters, n_clusters))
for index_cluster_1 in list(range(0, n_clusters)):
    for index_cluster_2 in list(range(index_cluster_1, n_clusters)):
        distance = round(compute_distance(cluster_geometric_sampler, index_cluster_1+1, index_cluster_2+1), 1)        
        graph_spatial_distance[index_cluster_1][index_cluster_2] = distance
        if distance!=0: values.append(distance)
plt.figure(figsize=(8, 6))
plt.title('Distribution of inter-distance')
plt.hist(values, edgecolor='k', bins=10)
plt.show()
print(np.matrix(graph_spatial_distance))
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
plt.figure(figsize=(6, 4))
plt.title('Distribution of centroid color distance')
plt.hist(values, bins=20, edgecolor='k')
plt.grid()
plt.show()
print(np.matrix(graph_color_distance))
# Arbitrary definition of tresholds
treshold_spatial_distance = 5
treshold_color_distance = 0.40
# Connected components
rule_spatial_union = graph_spatial_distance < treshold_spatial_distance
rule_color_union = graph_color_distance < treshold_color_distance
# Boolean AND 
union = rule_spatial_union * rule_color_union
graph = union.T * union
print(graph)
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
uf = UnionFind(n_clusters+1)
for i in range(0, n_clusters):
    for j in list(np.where(union[i][i+1:])[0]+i):
        uf.union(i+1, j+2)
print('rank:', uf.rank)
print('up:  ', uf.up)
from collections import defaultdict
components = defaultdict(list)
for i in range(n_clusters):
    components[uf.find(i)].append(i)
for component in list(components.values()):
    print(component)
for i, connect_components in enumerate(components.values()):
    if i != 0 and connect_components:
        print(connect_components)
        segmented_image = resized_assembly_image.copy()
        mask = np.ones((segmented_image.shape[0], segmented_image.shape[1]), dtype='bool')
        for cluster_id in connect_components:
            mask *= segments_fz != cluster_id
        segmented_image[mask] = [1, 1, 1]
        plt.imshow(segmented_image)
        plt.show()
background_path = '../input/lego/07-Background/'
background = background_path + np.random.choice(os.listdir(background_path))
back_image = plt.imread(background)
back_image = resize(back_image, (400, 300))
for i, connect_components in enumerate(components.values()):
    if i != 0 and connect_components:
        print(connect_components)
        segmented_image = resized_assembly_image.copy()
        mask = np.ones((segmented_image.shape[0], segmented_image.shape[1]), dtype='bool')
        for cluster_id in connect_components:
            mask *= segments_fz != cluster_id
        segmented_image[mask] = [0, 0, 0]
        
        back_image = plt.imread(background)
        back_image = resize(back_image, (400, 300))
        back_image[~mask] = [0, 0, 0]
        
        plt.imshow(segmented_image + back_image)
        plt.show()
segmented_image = resized_assembly_image.copy()
mask = np.ones((segmented_image.shape[0], segmented_image.shape[1]), dtype='bool')
for cluster_id in connect_components:
    mask *= segments_fz != cluster_id
pointer_y, pointer_x = np.where(mask == False)
random_index = np.random.choice(range(len(pointer_x)))
px = pointer_x[random_index]
py = pointer_y[random_index]
annotation = 'Piece id:{}  -  Confidence:{}'

x_annotate = 310
y_annotate = 10

plt.figure(figsize=(6, 6))
plt.imshow(resized_assembly_image)
plt.scatter(px, py, edgecolor='k', color='r', alpha=0.8)
plt.plot([px,x_annotate], [py,y_annotate], color='r', linewidth=0.5, alpha=0.8)
plt.annotate(annotation.format('sample', 'sample'), (x_annotate, y_annotate))
plt.axis('off')
plt.show()
plt.figure(figsize=(6, 6))
plt.imshow(resized_assembly_image)
plt.axis('off')

annotation = 'Piece id:{}  -  Confidence:{}'

for i, connect_components in enumerate(components.values()):
    if i != 0 and connect_components:
        
        print(connect_components)
        segmented_image = resized_assembly_image.copy()
        mask = np.ones((segmented_image.shape[0], segmented_image.shape[1]), dtype='bool')
        for cluster_id in connect_components:
            mask *= segments_fz != cluster_id
            
        pointer_y, pointer_x = np.where(mask == False)
        random_index = np.random.choice(range(len(pointer_x)))
        px = pointer_x[random_index]
        py = pointer_y[random_index]
        
        x_annotate = 310
        y_annotate = i*50
        
        plt.scatter(px, py, edgecolor='k', color='r', alpha=0.8)
        plt.plot([px,x_annotate], [py,y_annotate], linewidth=1, alpha=0.8)
        plt.annotate(annotation.format('sample', 'sample'), (x_annotate, y_annotate))

plt.show()