from collections import Counter
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision import transforms as T
from torchvision.utils import make_grid
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors


%matplotlib inline
def show_cluster(cluster, labels, dataset, limit=32):
    images = []
    labels = np.array(labels)
    indices = np.where(labels==cluster)[0]
    
    if not indices.size:
        print(f'cluster: {cluster} is empty.')
        return None
    
    for i in indices[:limit]:
        image, _ = dataset[i]
        images.append(image)
        
    gridded = make_grid(images)
    plt.figure(figsize=(15, 10))
    plt.title(f'cluster: {cluster}')
    plt.imshow(gridded.permute(1, 2, 0))
    plt.axis('off')
    
    
def show_neighbors(neighbors, dataset):
    images = []
    for n in neighbors:
        images.append(dataset[n][0])

    gridded = make_grid(images)
    plt.figure(figsize=(15, 10))
    plt.title(f'image and nearest neighbors')
    plt.imshow(gridded.permute(1, 2, 0))
    
    
def extract_features(model, dataset, batch_size=32):
    """
    Gets the output of a pytorch model given a dataset.
    """
    loader = DataLoader(dataset, batch_size=batch_size)
    features = []
    for image, _ in tqdm(loader, desc='extracting features'):
        output = model(Variable(image).cuda())
        features.append(output.data.cpu())
    return torch.cat(features).numpy() 
class FoodDataset(Dataset):
    def __init__(self, root, transforms=None, labels=[], limit=None):
        self.root = Path(root)
        self.image_paths = list(Path(root).glob('*/*.jpg'))
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.labels = labels
        self.transforms = transforms
        self.classes = set([path.parts[-2] for path in self.image_paths])
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index] if self.labels else 0
        image = Image.open(image_path)
        if self.transforms:
            return self.transforms(image), label
        return image, label
            
    def __len__(self):
        return len(self.image_paths)    
    
transforms = T.Compose([T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor()])
# data
root = '../input/images'
limit_images = 10000

# clustering
pca_dim = 50
kmeans_clusters = 100

# convnet
batch_size = 64
num_classes = 100
num_epochs = 2

dataset = FoodDataset(root=root, limit=limit_images)
dataset.classes
image, _ = dataset[9000]
image
# load resnet and alter last layer
model = resnet18()
model.fc = nn.Linear(512, num_classes)
model.cuda();

pca = IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
kmeans = MiniBatchKMeans(n_clusters=kmeans_clusters, batch_size=512, init_size=3*kmeans_clusters)
optimizer = Adam(model.parameters())
def cluster(pca, kmeans, model, dataset, batch_size, return_features=False):
    features = extract_features(model, dataset, batch_size)  
    reduced = pca.fit_transform(features)
    pseudo_labels = list(kmeans.fit_predict(reduced))
    if return_features:
        return pseudo_labels, features
    return pseudo_labels
def train_epoch(model, optimizer, train_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0
    pbar = tqdm(train_loader)
    for batch, (images, labels) in enumerate(pbar):
        optimizer.zero_grad()
        images = Variable(images).cuda()
        labels = Variable(labels).cuda().long()
        out = model(images)
        loss = F.cross_entropy(out, labels)
        total_loss += loss.data[0]
        pbar.set_description(f'training - loss: {total_loss / (batch + 1)}')
        loss.backward()
        optimizer.step()
raw_dataset = FoodDataset(root=root, transforms=transforms, limit=limit_images)
pseudo_labels, features = cluster(pca, kmeans, model, raw_dataset, batch_size, return_features=True)
plt.hist(pseudo_labels, bins=kmeans_clusters)
plt.title('cluster membership counts');
raw_dataset.classes ## all food types we have sampled
counts = Counter(pseudo_labels)
show_cluster(counts.most_common()[0][0], pseudo_labels, raw_dataset)
show_cluster(counts.most_common()[1][0], pseudo_labels, raw_dataset)
knn = NearestNeighbors(metric='cosine')
knn.fit(features)
anchor_image = 0
neighbors = knn.kneighbors([features[anchor_image]], n_neighbors=4, return_distance=False)[0]
show_neighbors(neighbors, raw_dataset)
for i in range(num_epochs):
    pseudo_labels = cluster(pca, kmeans, model, raw_dataset, batch_size) # generate labels
    labeled_dataset = FoodDataset(root=root, labels=pseudo_labels, transforms=transforms, limit=limit_images) # make new dataset with labels matched to images
    train_epoch(model, optimizer, labeled_dataset, batch_size) # train for one epoch
pseudo_labels, features = cluster(pca, kmeans, model, raw_dataset, batch_size, return_features=True)

plt.hist(pseudo_labels, bins=kmeans_clusters)
plt.title('cluster membership counts');
counts = Counter(pseudo_labels)
show_cluster(counts.most_common()[0][0], pseudo_labels, raw_dataset)
show_cluster(counts.most_common()[1][0], pseudo_labels, raw_dataset)
knn = NearestNeighbors(metric='cosine')
knn.fit(features)
anchor_image = 0
neighbors = knn.kneighbors([features[anchor_image]], n_neighbors=4, return_distance=False)[0]
show_neighbors(neighbors, raw_dataset)
for i in range(4):
    pseudo_labels = cluster(pca, kmeans, model, raw_dataset, batch_size) # generate labels
    labeled_dataset = FoodDataset(root=root, labels=pseudo_labels, transforms=transforms, limit=limit_images) # make new dataset with labels matched to images
    train_epoch(model, optimizer, labeled_dataset, batch_size) # train for one epoch
features = extract_features(model, raw_dataset, batch_size)  
knn = NearestNeighbors(metric='cosine')
knn.fit(features)
anchor_image = 0
neighbors = knn.kneighbors([features[anchor_image]], n_neighbors=4, return_distance=False)[0]
show_neighbors(neighbors, raw_dataset)
