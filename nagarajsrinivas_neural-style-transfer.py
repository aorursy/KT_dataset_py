import matplotlib.pyplot as plt

import sys



import torch

from torch import nn

from torch import optim

import torch.nn.functional as F

from torch.autograd import Variable

from PIL import Image

from torchvision import datasets, transforms, models
import numpy as np

%matplotlib inline
import os
os.listdir('../input')
#root = ''
model = models.vgg19(pretrained=True).features      # Only convolutional and maxpool layers
model
# We are freezing all vgg parameters since we are only optimising the target image

# Setting requires_grad = False ensures that none of the weights are changed. 

for param in model.parameters():

    param.requires_grad_(False)

    

# Now, the model has become a fixed feature extractor. 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model.to(device)
device
def load_image(img_path, max_size=1200, shape=None):

    ''' Load in and transform an image, making sure the image

       is <= 400 pixels in the x-y dims.'''

    if "http" in img_path:

        response = requests.get(img_path)

        image = Image.open(BytesIO(response.content)).convert('RGB')

    else:

        image = Image.open(img_path).convert('RGB')

    

    # large images will slow down processing

    if max(image.size) > max_size:

        size = max_size

    else:

        size = max(image.size)

    

    if shape is not None:

        size = shape

        

    in_transform = transforms.Compose([

                        transforms.Resize(size),

                        transforms.ToTensor(),

                        transforms.Normalize((0.485, 0.456, 0.406), 

                                             (0.229, 0.224, 0.225))])



    # discard the transparent, alpha channel (that's the :3) and add the batch dimension

    image = in_transform(image)[:3,:,:].unsqueeze(0)         # unsqueeze adds the extra dimension

    

    return image
# load in content and style image

content = load_image('../input/tarzan007/images_anim.jpg').to(device)
content.shape           # shape index --> -4, -3, -2, -1
content.shape[-2:]           # index from -2 to -1/0
# Resize style to match content, makes computation easier

style = load_image('../input/mydata/images_abstract.jpg', shape=content.shape[-2:]).to(device)
# To display a image which is of type tensor, we first have to convert it to numpy. 

def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()

    image = image.numpy().squeeze()             # 4-dimension to 3-dimension

    image = image.transpose(1,2,0)              

    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))           # De-normalizing

    #image = image.clip(0, 1)



    return image
# display the images

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# content and style images side-by-side

ax1.imshow(im_convert(content))

ax2.imshow(im_convert(style))
def get_features(image, vgg, layers=None):

    # Forward pass of the imamge through the model. 

    if layers is None:

        layers = {'0': 'conv1_1',

                  '5': 'conv2_1', 

                  '10': 'conv3_1', 

                  '19': 'conv4_1',

                  '21': 'conv4_2',             # content representation

                  '28': 'conv5_1'}

        

    features = {}        # This is also a dictionary

    x = image

    # model._modules is a dictionary holding each module(layer) in the model  

    # key : layer name, value : layer details

    for name, layer in vgg._modules.items():

        x = layer(x)                             # For the first iteration, the image is passed through layer 1, x then holds 

                                                 # the output of that layer(feature map) which is passed on to the next layer.

        # The image is passed through all the layers, but we extract the feature map only from the desired layers.

        if name in layers:

            features[layers[name]] = x           # layers[0] = conv1_1, features[conv1_1] = That layer's output feature map

            

    return features
#model._modules.items()
#for name, layer in model._modules.items():

#    print(name)

#    print(layer)
def gram_matrix(tensor):

    

    size = tensor.shape

    tensor = tensor.view(size[1],size[2]*size[3])   # Converting each feature map to a vector. Since we also have depth, this 

                                                    # will be depth * (h*w) matrix.

    tensor_transpose = tensor.t()

    

    gram = torch.mm(tensor, tensor_transpose) # This effectively mmultiplies all the features and gets the correlations

    

    return gram
# We extract the content and style feature only once. These remain the same throughout the process. 

content_features = get_features(content,model)

style_features = get_features(style,model)
for layer in content_features:

    print(content_features[layer])
# calculating the gram matrices for each layer of our style representation

style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
# creating a output image which carries the style image. 

# we start the styling process from the content image. Hence, the output will be the content image before the style is applied.

# iterate until the desired style is achieved

output = content.clone().requires_grad_(True).to(device) 



# We can also start with a blank image, but we may get the output image that is too diverted from our content image. 
# weights for each style layer 

# weighting earlier layers more will result in *larger* style artifacts

# we are excluding `conv4_2` our content representation

style_weights = {'conv1_1': 1.0,

                 'conv2_1': 0.8,

                 'conv3_1': 0.4,

                 'conv4_1': 0.2,

                 'conv5_1': 0.1}



content_weight = 1e3 # alpha

style_weight = 1e2  # beta
show_every = 400





optimizer = optim.Adam([output], lr=0.003)

epochs = 3000  



for e in range(1, epochs+1):

    

    # get the features from the output image

    output_features = get_features(output, model)         # contains features from both style and content layers  

    

    # the content loss

    content_loss = torch.mean((output_features['conv4_2'] - content_features['conv4_2'])**2)

    

    # the style loss

    # initialize the style loss to 0

    style_loss = 0         

    

    for layer in style_weights:        # Only style layers

        output_feature = output_features[layer]  # "output" style representation for the layer

        output_gram = gram_matrix(output_feature)      # A particular layer feature. 

        _, d, h, w = output_feature.shape

        # get the "style" style representation

        style_gram = style_grams[layer]

        # the style loss for one layer, weighted appropriately

        layer_style_loss = style_weights[layer] * torch.mean((output_gram - style_gram)**2)

        # add to the style loss

        style_loss += layer_style_loss / (d * h * w)   # updating the style loss after calculating each layer style loss. 

        # (d*h*w) specifies the total number of values in a particular layer. 

        

    # calculate the total loss

    total_loss = content_weight * content_loss + style_weight * style_loss      # alpha*content_loss + beta*style_loss

    

    # updating the output image itself (not the usual weights that we were updating previously)

    optimizer.zero_grad()

    total_loss.backward()

    optimizer.step()

    

    # display intermediate images and print the loss

    if(e % show_every == 0):

        print('Total loss: ', total_loss.item())

        plt.imshow(im_convert(output))

        plt.show()
# display content and final, target image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(im_convert(content))

ax2.imshow(im_convert(output))
# Saving the output style image

import cv2

cv2.imwrite('../input/Style_transfer_10.jpg',im_convert(output))