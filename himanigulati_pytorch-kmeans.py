from urllib.request import urlretrieve
urlretrieve('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv', 'iris_data.csv')
df = pd.read_csv('iris_data.csv')
df
df.info()
!pip install kmeans-pytorch
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
from kmeans_pytorch import kmeans, kmeans_predict
import torch.utils.data as data_utils

#ImageFolder class for using directory structure.
#from torchvision.datasets import ImageFolder
#from torchvision.transforms import ToTensor
df = pd.read_csv('iris_data.csv')
target_df = df[['species']]
df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
df_tensor = torch.tensor(df.to_numpy())
#target_df = torch.tensor(target_df.to_numpy())

df_tensor

#I have inherited torch.utils.data module as data_utils

class iris_dataset(data_utils.Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    
    def __float__(self):
        return 0.0
    #float(self.X_data), int(self.y_data)
dataset = iris_dataset(df_tensor, target_df)
dataset[0:5]
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
num_clusters = 3
cluster_ids_x, cluster_centers = kmeans(
    X=df_tensor, num_clusters=num_clusters, distance='euclidean', device=device
)

print(cluster_ids_x)
print(cluster_centers)
y = df_tensor
y
cluster_ids_y = kmeans_predict(
    y, cluster_centers, 'euclidean', device=device
)
print(cluster_ids_y)
import seaborn as sns
# Visualising the clusters - On the first two columns
sns.set_style("darkgrid")
plt.figure(figsize = (16,8))
plt.scatter(df_tensor[cluster_ids_y == 0, 0], df_tensor[cluster_ids_y == 0, 1], 
            s = 100, c = 'red', label = 'setosa')
plt.scatter(df_tensor[cluster_ids_y == 1, 0], df_tensor[cluster_ids_y == 1, 1], 
            s = 100, c = 'blue', label = 'versicolour')
plt.scatter(df_tensor[cluster_ids_y == 2, 0], df_tensor[cluster_ids_y == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(cluster_centers[:, 0], cluster_centers[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()
project_name = 'pytorch_kmeans'
!pip install jovian --upgrade --quiet
import jovian
jovian.commit(project=project_name)

