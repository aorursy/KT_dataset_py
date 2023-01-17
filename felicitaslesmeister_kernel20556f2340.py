!pip install jovian --upgrade --quiet
project_name = "course-project-apples"
DATA_DIR = "../input/fruits/fruits-360/Training"
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
batch_size = 50
train_ds = ImageFolder(DATA_DIR, transform=ToTensor())
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
for images, _ in train_dl:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
import jovian
jovian.commit(project=project_name)
