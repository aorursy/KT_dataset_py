import os

import time 

from tqdm.notebook import tqdm



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torchvision import models, utils, transforms



from glob import glob

from PIL import Image
data_path = os.path.join("/kaggle", "input", "style-transfer")

content_path = os.path.join(data_path, "content")

style_path = os.path.join(data_path, "style")

content_fps = sorted(glob(os.path.join(content_path, "*")))

style_fps = sorted(glob(os.path.join(style_path, "*")))

print(content_fps)

print(style_fps)
transform = transforms.Compose([

    transforms.Resize((512, 512)),

    transforms.ToTensor(),

    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

])



inverse_transform = transforms.Compose([

    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))

])
content_images = []

for content_fp in content_fps:

    image = Image.open(content_fp) 

    image = transform(image) # Size([3, 512, 512])

    content_images.append(image)

    

style_images = []

for style_fp in style_fps:

    image = Image.open(style_fp) 

    image = transform(image) # Size([3, 512, 512])

    style_images.append(image)
def get_grid(images):



    grid = utils.make_grid(images)

    grid = grid.transpose(0, 2).transpose(0 ,1)



    return grid
content_grid = get_grid(content_images)

style_grid = get_grid(style_images)

fig, axes = plt.subplots(2, 1, figsize=(20, 10))

axes[0].imshow(content_grid)

axes[0].set_title("Content Images Normalized")

axes[1].imshow(style_grid)

axes[1].set_title("Style Images Normalized")

plt.show()
content_images_raw = []

for image in content_images:

    image_raw = inverse_transform(image)

    content_images_raw.append(image_raw)

    

style_images_raw = []

for image in style_images:

    image_raw = inverse_transform(image)

    style_images_raw.append(image_raw)
content_grid = get_grid(content_images_raw)

style_grid = get_grid(style_images_raw)

fig, axes = plt.subplots(2, 1, figsize=(20, 10))

axes[0].imshow(content_grid)

axes[0].set_title("Content Images")

axes[1].imshow(style_grid)

axes[1].set_title("Style Images")

plt.show()
vgg = models.vgg19(pretrained=True)
for p in vgg.parameters():

    p.requires_grad_(False)
vgg
for i, feature in enumerate(vgg.features):

    if isinstance(feature, nn.MaxPool2d):

        vgg.features[i] = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)
vgg.features
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
vgg = vgg.to(device).eval()
content_layers = ['conv4_2']

style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']



id_to_layer = {

    '0': 'conv1_1', 

    '5': 'conv2_1', 

    '10': 'conv3_1', 

    '19': 'conv4_1',

    '21': 'conv4_2',  

    '28': 'conv5_1'

}

layer_to_id = {v:k for k, v in id_to_layer.items()}

print(id_to_layer)

print(layer_to_id)



content_layers_id = [int(layer_to_id[l]) for l in content_layers]

style_layers_id = [int(layer_to_id[l]) for l in style_layers]

print(content_layers_id)

print(style_layers_id)
def extract_content_features(image):

    """

    image -- ImageTensor of Size([n_c, n_h, n_w])

    """

    X = image.unsqueeze(0)

    content_features = []

    for i, feature in enumerate(vgg.features):

        X = feature(X)

        if i in content_layers_id:

            content_feature = X.view(-1)

            content_features.append(content_feature)

        

    return content_features
def extract_style_features(image):

    """

    image -- ImageTensor of Size([n_c, n_h, n_w])

    """

    X = image.unsqueeze(0)

    style_features = []

    for i, feature in enumerate(vgg.features):

        X = feature(X)

        if i in style_layers_id:

            out = X.view(X.size(1), -1)

            style_feature = torch.mm(out.clone(), out.t().clone())

            style_feature = style_feature.view(-1)

            style_feature = style_feature / len(style_feature)

            style_features.append(style_feature)

        

    return style_features
def compute_content_loss(content_features_c, content_features_g, weights):

    

    weights = torch.Tensor(weights)

    weights = weights / weights.sum()

    content_loss = 0

    for feature_c, feature_g, weight in zip(content_features_c, content_features_g, weights):

        loss = nn.MSELoss(reduction="mean")(feature_c, feature_g)

        loss = loss * weight

        content_loss = content_loss + loss

    content_loss = content_loss / len(weights)

    

    return content_loss
def compute_style_loss(style_features_c, style_features_g, weights):

    

    weights = torch.Tensor(weights)

    weights = weights / weights.sum()

    style_loss = 0

    for feature_c, feature_g, weight in zip(style_features_c, style_features_g, weights):

        loss = nn.MSELoss(reduction="mean")(feature_c, feature_g)

        loss = loss * weight

        style_loss = style_loss + loss

    style_loss = style_loss / len(weights)

    

    return style_loss
def compute_loss(content_loss, style_loss, content_weight, style_weight):

    

    return (content_weight * content_loss) + (style_weight * style_loss)
def generate_image(content_image, style_image, content_weight=100, style_weight=1, epochs=200, lr=0.1, print_steps=1, display_plot=True):

    

    print_every = epochs // print_steps

    

    weight_sum = content_weight + style_weight

    content_weight /= weight_sum

    style_weight /= weight_sum

    print("Weight[content]:{}    Weight[style]:{}".format(content_weight, style_weight))

    

    content_features_c = extract_content_features(content_image.to(device))

    style_features_s = extract_style_features(style_image.to(device))

    

    generated_image = torch.randn((3, 512, 512)).to(device).requires_grad_(True)

    optimizer = optim.Adam([generated_image], lr=lr)

    

    content_losses, style_losses, losses = [], [], []

    for epoch in tqdm(range(1, epochs+1)):

        

        optimizer.zero_grad()

        

        content_features_g = extract_content_features(generated_image.to(device))

        content_loss = compute_content_loss(content_features_c, content_features_g, [1])

        

        style_features_g = extract_style_features(generated_image.to(device))

        style_loss = compute_style_loss(style_features_s, style_features_g, [0.75, 0.5, 0.2, 0.2, 0.2])

        

        loss = compute_loss(content_loss, style_loss, content_weight, style_weight)

        loss.backward()

        

        optimizer.step()

        

        content_losses.append(content_loss.item())

        style_losses.append(style_loss.item())

        losses.append(loss.item())

        

        if epoch == 1 or epoch % print_every == 0:

            message = "Epoch:{}    Loss:{}    ContentLoss:{}    StyleLoss:{}".format(epoch,

                                                                                    loss.item(),

                                                                                    content_loss.item(),

                                                                                    style_loss.item())

            print(message)

            

            

    if display_plot:

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(losses)

        axes[0].set_title("Loss")

        axes[1].plot(content_losses)

        axes[1].set_title("Content Loss")

        axes[2].plot(style_losses)

        axes[2].set_title("Style Loss")

    

    images = [content_image, style_image, generated_image]

    images = [inverse_transform(image).detach().to("cpu") for image in images]

    grid = get_grid(images)

    plt.figure(figsize=(20, 5))

    plt.imshow(grid)

    plt.title("Content Image;    Style Image;    Generated Image")

    

    return plt.show()
generate_image(content_images[0], style_images[0])
generate_image(content_images[0], style_images[3], display_plot=False)
generate_image(content_images[1], style_images[3], display_plot=False)
generate_image(content_images[1], style_images[4], display_plot=False)
generate_image(content_images[2], style_images[0], display_plot=False)
generate_image(content_images[2], style_images[3], display_plot=False)
generate_image(content_images[3], style_images[1], display_plot=False)
generate_image(content_images[3], style_images[2], display_plot=False)
generate_image(content_images[4], style_images[0], display_plot=False)
generate_image(content_images[4], style_images[4], display_plot=False)