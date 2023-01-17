import time
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import albumentations as A


class DogDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = pd.read_csv('../input/dog-breed-identification/labels.csv')
        self.transform = transform
        
        breeds=list(self.img_list['breed'].unique())
        self.breed2idx = {b: i for i, b in enumerate(breeds)}

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_row = self.img_list.iloc[idx]
        image = cv2.imread('../input/dog-breed-identification/train/' + img_row['id'] + '.jpg')
        label = self.breed2idx[img_row['breed']]

        if self.transform is not None:
            image = self.transform(image=image)
        image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, label

transform = A.Compose([A.RandomResizedCrop(height=224, width=224, p=1),
                       A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.MotionBlur(blur_limit=3, p=1),
                       A.Rotate(limit=45, p=1),
                       A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True, p=1.0)])

data_loader = DataLoader(DogDataset(transform=transform), batch_size=64, shuffle=True, num_workers=2)
%%timeit -r 2 -n 5
opencv_alb_times = []
start_time = time.time()
for image, label in data_loader:
    image = image.cuda()
    label = label.cuda()
    pass
opencv_alb_time = time.time() - start_time
opencv_alb_times.append(opencv_alb_time)
print(str(opencv_alb_time) + ' sec')
!apt-get install libturbojpeg
!pip install jpeg4py
import time
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import albumentations as A
import jpeg4py as jpeg


class DogDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = pd.read_csv('../input/dog-breed-identification/labels.csv')
        self.transform = transform
        
        breeds=list(self.img_list['breed'].unique())
        self.breed2idx = {b: i for i, b in enumerate(breeds)}

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_row = self.img_list.iloc[idx]
        image = jpeg.JPEG('../input/dog-breed-identification/train/' + img_row['id'] + '.jpg').decode()
        label = self.breed2idx[img_row['breed']]

        if self.transform is not None:
            image = self.transform(image=image)
        image = torch.from_numpy(image['image'].transpose(2, 0, 1))
        return image, label

transform = A.Compose([A.RandomResizedCrop(height=224, width=224, p=1),
                       A.HorizontalFlip(p=0.5),
                       A.VerticalFlip(p=0.5),
                       A.MotionBlur(blur_limit=3, p=1),
                       A.Rotate(limit=45, p=1),
                       A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=True, p=1.0)])

data_loader = DataLoader(DogDataset(transform=transform), batch_size=64, shuffle=True, num_workers=2)
%%timeit -r 2 -n 5
jpeg4py_alb_times = []
start_time = time.time()
for image, label in data_loader:
    image = image.cuda()
    label = label.cuda()
    pass
jpeg4py_alb_time = time.time() - start_time
jpeg4py_alb_times.append(jpeg4py_alb_time)
print(str(jpeg4py_alb_time) + ' sec')
import time
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import jpeg4py as jpeg

import albumentations as A
import kornia.augmentation as K
import torch.nn as nn


class DogDataset(Dataset):
    def __init__(self, transform=None):
        self.img_list = pd.read_csv('../input/dog-breed-identification/labels.csv')
        self.transform = transform
        
        breeds=list(self.img_list['breed'].unique())
        self.breed2idx = {b: i for i, b in enumerate(breeds)}

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_row = self.img_list.iloc[idx]
        image = jpeg.JPEG('../input/dog-breed-identification/train/' + img_row['id'] + '.jpg').decode()
        label = self.breed2idx[img_row['breed']]

        if self.transform is not None:
            image = self.transform(image=image)
        image = torch.from_numpy(image['image'].transpose(2, 0, 1).astype(np.float32))
        return image, label

alb_transform = A.Compose([A.RandomResizedCrop(height=224, width=224, p=1)])

mean_std = torch.Tensor([0.5, 0.5, 0.5])*255
kornia_transform = nn.Sequential(
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),
    K.RandomMotionBlur(3, 35., 0.5),
    K.RandomRotation(degrees=45.0),
    K.Normalize(mean=mean_std,std=mean_std)
)

data_loader = DataLoader(DogDataset(transform=alb_transform), batch_size=64, shuffle=True, num_workers=2)
%%timeit -r 2 -n 5
jpeg4py_kornia_times = []
start_time = time.time()
for image, label in data_loader:
    image = kornia_transform(image.cuda())
    label = label.cuda()
    pass
jpeg4py_kornia_time = time.time() - start_time
jpeg4py_kornia_times.append(jpeg4py_kornia_time)
print(str(jpeg4py_kornia_time) + ' sec')
!pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100
import time
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import kornia.augmentation as K
import torch.nn as nn
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
class DALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.img_list = pd.read_csv('../input/dog-breed-identification/labels.csv')

        breeds=list(self.img_list['breed'].unique())
        self.breed2idx = {b: i for i, b in enumerate(breeds)}
        
        self.img_list['label'] = self.img_list['breed'].map(self.breed2idx)
        self.img_list['data'] = '../input/dog-breed-identification/train/' + self.img_list['id'] + '.jpg'
        
        self.img_list[['data', 'label']].to_csv('dali.txt', header=False, index=False, sep=' ')
        
        self.input = ops.FileReader(file_root='.', file_list='dali.txt')
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.DALIImageType.RGB)
        #self.decode = ops.ImageDecoderRandomCrop(device = "mixed", output_type = types.DALIImageType.RGB)
        self.resize = ops.RandomResizedCrop(device = "gpu", size=(224, 224))
        self.transpose = ops.Transpose(device='gpu', perm = [2, 0, 1])
        self.cast = ops.Cast(device='gpu', dtype=types.DALIDataType.FLOAT)

    def define_graph(self):
        images, labels = self.input(name="Reader")
        images = self.decode(images)
        images = self.resize(images)
        images = self.cast(images)
        output = self.transpose(images)
        return (output, labels)
def DALIDataLoader(batch_size):
    num_gpus = 1
    pipes = [DALIPipeline(batch_size=batch_size, num_threads=2, device_id=device_id) for device_id in range(num_gpus)]

    pipes[0].build()
    dali_iter = DALIGenericIterator(pipelines=pipes, output_map=['data', 'label'], 
                                    size=pipes[0].epoch_size("Reader"), reader_name=None, 
                                    auto_reset=True, fill_last_batch=True, dynamic_shape=False, 
                                    last_batch_padded=True)
    return dali_iter

data_loader = DALIDataLoader(batch_size=64)

mean_std = torch.Tensor([0.5, 0.5, 0.5])*255
kornia_transform = nn.Sequential(
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),
    K.RandomMotionBlur(3, 35., 0.5),
    K.RandomRotation(degrees=45.0),
    K.Normalize(mean=mean_std,std=mean_std)
)
%%timeit -r 2 -n 5
dali_kornia_times = []
start_time = time.time()
for feed in data_loader:
    # image is already on GPU
    image = kornia_transform(feed[0]['data'])
    label = feed[0]['label'].cuda()
    pass
dali_kornia_time = time.time() - start_time
dali_kornia_times.append(dali_kornia_time)
print(str(dali_kornia_time) + ' sec')
import numpy as np
import matplotlib.pyplot as plt
 
left = np.array([1, 2, 3, 4])
height = np.array([71, 41.5, 26.2, 8.1])
label = ["OpenCV\n+\nAlbumentations", "jpeg4py\n+\nAlbumentations", "jpeg4py\n+\nKornia", "NVIDIA DALI\n+\nKornia"]
plt.bar(left, height, tick_label=label, align="center")