import math

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn

class DiabeteDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
dataset = datasets.MNIST(".",
                train=True,
                download=True,
              )

image, label = next(iter(dataset))
print(image)
print(label)
image
dataset = datasets.MNIST(".",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(14)]
                )
              )

# Check one sample
image, label = next(iter(dataset))
print(image)
print(label)
image
dataset = datasets.MNIST(".",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(14), transforms.ToTensor()]
                )
              )

# Check one sample
image, label = next(iter(dataset))
print(type(image))
print(image.size())
print(label)

plt.style.use("ggplot")
plt.hist(data[0].flatten())
plt.axvline(data[0].mean())
plt.show()

dataset = datasets.MNIST(".",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(28), transforms.ToTensor()]
                )
              )

imgs_lbls = list(iter(dataset))
imgs = [sample[0] for sample in imgs_lbls]
imgs_tensor = torch.Tensor(len(imgs), *(imgs[0].size()))
torch.cat(imgs, out=imgs_tensor)
print('Training imgs', imgs_tensor.size())


mean, std = imgs_tensor.mean(), imgs_tensor.std()
mean, std
# # dataloader could be used easily to compute mean and std 
# # It is especially useful if dataset is too large, then we can divide it into many batches for aggregating multiple times
# dataloader = DataLoader(dataset,
#                         batch_size=len(dataset),
#                         shuffle=True)
# data = next(iter(dataloader))
# data[0].mean(), data[0].std()
dataset = datasets.MNIST(".",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize(mean, std)]
                )
              )


imgs_lbls = list(iter(dataset))
imgs = [sample[0] for sample in imgs_lbls]
imgs_tensor = torch.Tensor(len(imgs), *(imgs[0].size()))
torch.cat(imgs, out=imgs_tensor)
print('Training imgs', imgs_tensor.size())

plt.style.use("ggplot")
plt.hist(data[0].flatten())
plt.axvline(data[0].mean())
plt.show()


mean, std = imgs_tensor.mean(), imgs_tensor.std()
mean, std


dataloader = DataLoader(dataset,
                        batch_size=64,
                        shuffle=True)