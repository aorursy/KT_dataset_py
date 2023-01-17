from __future__ import print_function



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

import torchvision.transforms as transforms

import torchvision.models as models



import copy
#Image Loader Function 

def image_loader(img_path, max_size=600, shape=None):

    image = Image.open(img_path).convert('RGB')

    

    if max(image.size) > max_size:

        imgsize = max_size

    else:

        imgsize = max(image.size)

        

    if shape is not None:

        size = shape

        

    in_transform = transforms.Compose([

        transforms.Resize((imgsize, int(1.5*imgsize))),

        transforms.ToTensor(), #convert to tensor

        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    

    return image
#Load Style Image

style = image_loader("../input/picadylan/pics.jpg")
def im_show(tensor):  #function to show image

    image = tensor.to("cpu").clone().detach() #clone to not do changes on it

    image = image.numpy().squeeze() #remove fake batch dimension

    image = image.transpose(1, 2, 0)

    image = image * np.array((0.229, 0.224, 0.225)) + np.array(

    (0.485, 0.456, 0.406))

    image = image.clip(0, 1)

    

    return image
#Feature Extraction Function

def get_features(image, model, layers=None):

    if layers is None:

        layers = {'0': 'conv1_1','5': 'conv2_1',

                  '10': 'conv3_1',

                  '19': 'conv4_1',

                  '21': 'conv4_2',  ## content layer

                  '28': 'conv5_1'}

        

    features = {}

    x = image

    for name, layer in enumerate(model.features):

        x = layer(x)

        if str(name) in layers:

            features[layers[str(name)]] = x

            

    return features
#gram marix function

#gram matrix = mat * tranpose(mat)



def gram_matrix(tensor):

    _,n_filters,h,w = tensor.size() #abs(=1)

    #b = number of feature maps

    #(c,d) = dimensions of a f. map (N=c*d)

    tensor = tensor.view(n_filters, h * w)

    G = torch.mm(tensor, tensor.t()) #gram matrix

    return G
cnn = models.vgg19(pretrained=True)
for param in cnn.parameters():

    param.requires_grad_(False)
#AvgPool2d instead of MaxPool2d for better results

for i, layer in enumerate(cnn.features):

    if isinstance(layer, torch.nn.MaxPool2d):

        cnn.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#Check CUDA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn.to(device).eval()
#load content image

content = image_loader("../input/picadylan/dylan.jpg").to(device)

style = style.to(device)



content_features = get_features(content, cnn)

style_features = get_features(style, cnn)
style_grams = {

    layer: gram_matrix(style_features[layer]) for layer in style_features}
#target image

target = content.clone().requires_grad_(True).to(device) #clone content image for target

#for target with random white noise use the line below

#target = torch.randn_like(content).requires_grad_(True).to(device)
#Style weights for different layers

style_weights = {'conv1_1': 0.75,

                 'conv2_1': 0.5,

                 'conv3_1': 0.2,

                 'conv4_1': 0.2,

                 'conv5_1': 0.2}
#default weights

content_weight = 1e4

style_weight = 1e2
#optimizer LBFGS

optimizer = optim.LBFGS([target.requires_grad_()])

num_iterations = 400
#Style Transfer Function

def styl_trans():

    i = [0]

    while i[0] <= num_iterations:

        def closure():

            optimizer.zero_grad()

            target_features = get_features(target, cnn)

    

            content_loss = torch.mean((target_features['conv4_2'] -

                             content_features['conv4_2']) ** 2)

  

            style_loss = 0

            for layer in style_weights:

                target_feature = target_features[layer]

                target_gram = gram_matrix(target_feature)

                _, d, h, w = target_feature.shape

                style_gram = style_grams[layer]

                layer_style_loss = style_weights[layer] * torch.mean(

                    (target_gram - style_gram) ** 2)

                style_loss += layer_style_loss / (d * h * w)

    

                style_score = style_weight * style_loss

                content_score = content_weight * content_loss

                total_loss = content_weight * content_loss + style_weight * style_loss

                total_loss.backward(retain_graph=True)

        

            i[0] += 1

            if i[0] % 50 == 0:

                content_fraction = round(

                    content_weight*content_loss.item()/total_loss.item(), 2)

                style_fraction = round(

                    style_weight*style_loss.item()/total_loss.item(), 2)

                print('Iteration {}, (content-loss: {}, style-loss {})'.format(

                    i, content_fraction, style_fraction))

            return style_score + content_score

        optimizer.step(closure)

    final_img = im_show(target)

    return final_img

#OUTPUT

output = styl_trans()

fig = plt.figure()

plt.imshow(output)

plt.axis('off')