import torch
from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from PIL import Image
import torch.nn.functional as F

import scipy.ndimage.morphology as morph
path = "../input/pascal-voc-2012/VOC2012/ImageSets/Segmentation/train.txt"
 
#print(os.listdir("../input/pascal-voc-2012/VOC2012/JPEGImages"))
f = open(path, "r").read().split('\n')
f = f[:1464]
folder_data = "../input/pascal-voc-2012/VOC2012/JPEGImages"
folder_mask = "../input/pascal-voc-2012/VOC2012/SegmentationClass"
tfs = transforms.Compose([transforms.Resize((256, 256)),
                   transforms.ToTensor()])

img = Image.open(folder_data + "/" + f[21] + ".jpg").convert('RGB')
seg = Image.open(folder_mask + "/" + f[21] + ".png").convert('RGB')
resize = transforms.Resize((256, 256))
seg = resize(seg)
img = resize(img)
s = np.asarray(seg).tolist()
s[200][200]  == [0, 0, 0]
#seg
s[200][200]

seg
img
class Segdata(Dataset):
    def __init__(self):
        self.img_paths = os.listdir(folder_data)
        self.seg_paths = os.listdir(folder_mask)
        self.transform = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor()])
        #self.resize = transforms.Resize((256, 256))
        self.data = len(f)
    
    def __len__(self):
        return len(f)
    
    def __getitem__(self,idx):
        img = Image.open(folder_data + "/" + f[idx] + ".jpg").convert('RGB')
        img = self.transform(img)
        
        seg = Image.open(folder_mask + "/" + f[idx] + ".png").convert('RGB')
        seg = self.transform(seg)
        
        return img,seg
        
    
dataset = Segdata()
i,seg = dataset[6]
img = i.permute((1,2,0))
mask= seg.permute((1,2,0))
plt.imshow(img)
plt.imshow(mask)
!pip install -U segmentation-models-pytorch albumentations --user
import segmentation_models_pytorch as smp
ENCODER = 'resnet18'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'sigmoid'# could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda'

# create segmentation model with pretrained encoder,here FPN you can use UNET and others
model = smp.FPN(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
train_dataset = Segdata()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=10)
loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([ 
    dict(params=model.parameters(), lr=0.001),
])
train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)
max_score = 0

for i in range(0,15): #epochs = 15
    
    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
dataset = Segdata()
i,seg = dataset[10]
mask = seg.permute((1,2,0))
out= model.predict(i.to(DEVICE).unsqueeze(0))
print("Original mask image")
plt.imshow(mask)
plt.show()
print("predicted mask")
plt.imshow((out.squeeze().cpu().numpy().round()))