! wget https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt
import cv2
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.datasets.cifar as cifar
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image 

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

import json
with open("imagenet1000_clsidx_to_labels.txt") as f:
    idx2label = eval(f.read())

# Load any cat and dog
def tensor2image(tensor):
    return np.transpose(tensor.numpy(), (1,2,0))
def image2tensor(image):
    image = np.transpose(np.array(image), (2,0,1))
    return torch.from_numpy(np.array(image)).float()
cat = np.array(Image.open('/kaggle/input/dogs-cats-images/dog vs cat/dataset/test_set/cats/cat.4265.jpg'))/256
dog = np.array(Image.open('/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs/dog.155.jpg'))/256
# Load model
model = models.vgg16(pretrained=True,)

# Pass tensors for predition
catTensor = Variable(image2tensor(cat).unsqueeze_(0), requires_grad=True)
dogTensor = Variable(image2tensor(dog).unsqueeze_(0), requires_grad=True)
outCat = model(catTensor)
outDog = model(dogTensor)

# Get outputs 
values, indices = torch.max(outCat, 1)
cat_pred = idx2label[int(indices[0])]
values, indices = torch.max(outDog, 1)
dog_pred = idx2label[int(indices[0])]

# Plot'em
f, axarr = plt.subplots(2, figsize=(8,8))
f.suptitle('A Cat and a Dog', fontsize=16)
axarr[0].imshow(cat)
axarr[0].set_title("VGG Prediction: "+cat_pred)
axarr[1].imshow(dog)
axarr[1].set_title("VGG Prediction: "+dog_pred)
print(model)
class CamVGG(nn.Module):
    def __init__(self):
        super(CamVGG, self).__init__()
        self.vgg = models.vgg16(pretrained=True,)
        self.until_last_conv = self.vgg.features[:30] 
        # Recreate MaxPool Layer
        self.max_pool = self.vgg.features[30]
        # Get VGG Adaptive Pool 
        self.adaptive_pool = self.vgg.avgpool
        # Get VGG Classifier 
        self.classifier = self.vgg.classifier
        # Gradients
        self.gradients = None
    
    def hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.until_last_conv(x)
        h = x.register_hook(self.hook)
        x = self.max_pool(x)
        x = self.adaptive_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

CamModel = CamVGG()
pred = CamModel(catTensor)
pred_idx = int(pred.argmax(dim=1)[0])
cat_pred = idx2label[pred_idx]
# Propagate predicted class
pred[:, pred_idx].backward()
# Get the gradient
gradients = CamModel.gradients
# Take the mean on given dimentions... 
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
# Multiply calculated gradients by last convolution
activations = CamModel.until_last_conv(catTensor).detach()
for i in range(512):
    activations[:, i, :, :] *= pooled_gradients[i]
# Calculate mean for all convolutions to create heatmap
heatmap = torch.mean(activations, dim=1).squeeze()
# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= torch.max(heatmap)
# Plot Heatmap
plt.matshow(heatmap.squeeze())

# Get image as BGR
img = cat[...,::-1]*255
# Resize Heatmap and apply color map
heatmap1 = np.array(heatmap)
heatmap1 = cv2.resize(heatmap1, (img.shape[1], img.shape[0]))
heatmap1 = np.uint8(255 * heatmap1)
heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
# Superpose image
superimposed_img = heatmap1 * 0.4 + img
superimposed_img = superimposed_img[...,::-1]

plt.imshow(superimposed_img/255)
def getImageCamHeatmap(camModel, img):
    # Model pass
    imgTensor = Variable(image2tensor(img).unsqueeze_(0), requires_grad=True)
    pred = camModel(imgTensor)
    pred_idx = int(pred.argmax(dim=1)[0])
    prediction = idx2label[pred_idx]
    
    # Propagate predicted class
    pred[:, pred_idx].backward()
    # Get the gradient
    gradients = camModel.gradients
    # Take the mean on given dimentions... 
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # Multiply calculated gradients by last convolution
    activations = camModel.until_last_conv(imgTensor).detach()
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    # Calculate mean for all convolutions to create heatmap
    heatmap = torch.mean(activations, dim=1).squeeze()
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    # Plot Heatmap
    
    # Get image as BGR
    img = img[...,::-1]*255
    # Resize Heatmap and apply color map
    heatmap1 = np.array(heatmap)
    heatmap1 = cv2.resize(heatmap1, (img.shape[1], img.shape[0]))
    heatmap1 = np.uint8(255 * heatmap1)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    # Superpose image
    superimposed_img = heatmap1 * 0.4 + img
    superimposed_img = superimposed_img[...,::-1]
    
    return prediction, heatmap, superimposed_img

dog = np.array(Image.open('/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs/dog.566.jpg'))/256
pred, hmap, superposedImg = getImageCamHeatmap(CamModel, dog)
plt.title(pred)
plt.imshow(superposedImg/255)
dog = np.array(Image.open('/kaggle/input/dogs-cats-images/dog vs cat/dataset/training_set/dogs/dog.785.jpg'))/256
pred, hmap, superposedImg = getImageCamHeatmap(CamModel, dog)
plt.title(pred)
plt.imshow(superposedImg/255)
