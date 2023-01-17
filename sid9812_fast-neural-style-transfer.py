%matplotlib inline

import os
import sys
import time
import re
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import torch.onnx



size = (400,400)
img_transform = transforms.Compose([
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
data_dir = '../input/cocotest2014/test2014'
train_data = datasets.ImageFolder(data_dir, transform=img_transform)

batch_size = 1
num_workers = 0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)

print(len(train_loader))

dataiter = iter(train_loader)
images,_= dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(10):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
#     ax.set_title(classes[labels[idx]])



def load_image(img_path,shape = None):
  max_size = 400
  if "http" in img_path:
    response = requests.get(img_path)
    image = Image.open(BytesIO(response.content))
  else:
    image = Image.open(img_path).covert('RGB')

#   if max(image.size) > max_size:
#     size = max_size
#   else:
#     size = max(image.size)

#   if shape is not None:
#     size = shape

  im_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
  image = im_transform(image)[:3,:,:].unsqueeze(0)

  return image 
def img_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    image = image.clip(0, 1)

    return image
def get_features(image, model, layers = None):
  if layers is None:
    layers = {
        '0' : 'conv1_1',
        '1' : 'conv2_1',
        '2' : 'conv3_1',
        '3' : 'conv4_1',
        '4' : 'conv4_2',
        '5' : 'conv5_1'
    }
  features = {}
  x = image

  for name,layer in model._modules.items():
    x = layer(x)
    if name in layers:
      features[layers[name]] = x

  return features
# def gram_matrix(tensor):
#   b,d,h,w = tensor.size()
#   tensor = tensor.view(d,h*w)
#   gram = torch.mm(tensor,tensor.t())
#   return gram

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_image_path = "https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg"
style_image = load_image(style_image_path).to(device)
# style_image = style_image.repeat(batch_size, 1, 1, 1).to(device)
# plt.imshow(img_convert(style_image))
class TransformerNet(torch.nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
vgg = models.vgg19(pretrained=True).features

  # freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

transformer = TransformerNet().to(device)
optimizer = optim.Adam(transformer.parameters(), 0.01)
content_image_path = "https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg"
content_image = load_image(content_image_path).to(device)
plt.imshow(img_convert(style_image))
style_weights = {'conv1_1': 1.,
                  'conv2_1': 0.75,
                  'conv3_1': 0.2,
                  'conv4_1': 0.2,
                  'conv5_1': 0.2
                  }

content_weight = 1000  # alpha
style_weight = 1e8  # beta

style_features = get_features(normalize_batch(style_image),vgg)

style_grams = {}
for layer in style_features:
    style_grams[layer] = gram_matrix(style_features[layer])
n_epochs = 4

for epoch in range(1, n_epochs+1):
    transformer.train()
    agg_content_loss = 0.
    agg_style_loss = 0.
    count = 0
    
    for batch_i, (x, target) in enumerate(train_loader):
        n_batch = len(x)
        count += n_batch
        
        x = x.to(device)
        y = transformer(x)
        
        x = normalize_batch(x)
        y = normalize_batch(y)
        
        target_features = get_features(y,vgg)
        content_features = get_features(x,vgg)
        
        content_loss = torch.mean(torch.pow((target_features['conv4_2'] - content_features['conv4_2']),2))
        
        style_loss = 0.0
        
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _,d,h,w = target_feature.size()

            style_gram = style_grams[layer]
            layer_style_loss = (style_weights[layer])*torch.mean(torch.pow((target_gram - style_gram),2))

            style_loss += layer_style_loss
            
        total_loss = content_weight*content_loss + style_weight*style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        agg_content_loss += content_loss.item()
        agg_style_loss += style_loss.item()
        
        
        if (batch_i + 1) % 2000 == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch , count, len(train_data),
                                  agg_content_loss / (batch_i + 1),
                                  agg_style_loss / (batch_i + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_i + 1)
                )
                print(mesg)
                transformer.eval()
                img = transformer(content_image)
                plt.imshow(img_convert(img))
                plt.show()
                transformer.train()

    print('Epoch %d ' %
                (epoch ))
    transformer.eval()
    img = transformer(content_image)
    plt.imshow(img_convert(img))
    plt.show()
    transformer.train()
    
        
transformer.eval()
img = transformer(content_image)
plt.imshow(img_convert(img))
# plt.imshow(img_convert(content_image))
#     transformer.train()
