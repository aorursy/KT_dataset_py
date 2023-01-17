from PIL import Image

import numpy as np



import torch

import torch.optim as optim

from torchvision import transforms, models

import matplotlib.pyplot as plt

import requests
%matplotlib inline
vgg = models.vgg19(pretrained=True).features
# Freezing parameters

for param in vgg.parameters():

    param.requires_grad=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)
def load_image(img_path, max_size=400, shape=None):

    ''' Load in image and transform an image making sure the image is <= 400 pixels in x-y dims'''

    if "http" in img_path:

        response = requests.get(img_path)

        image = Image.open(BytesIO(response.content)).convert('RGB')

    else:

        image = Image.open(img_path).convert('RGB')

    

    # Large image will slow down processing

    if max(image.size) > max_size:

        size = max_size

    else:

        size = max(image.size)

    

    # Checking shape

    if shape is not None:

        size = shape

        

    # Transforming image

    in_transform = transforms.Compose([transforms.Resize(size),

                                     transforms.ToTensor(),

                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

                                     )

    # Discard the transparent, alpha channel that's (channel 3) and add batch dimension

    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image
# lOAD coontent and style image

content = load_image('../input/style-transfer-test/content.jpg').to(device)

# resize style to match content make code easier

style = load_image('../input/style-transfer-test/style.jpg', shape=content.shape[-2:]).to(device)
# helper function to convert tensor image to numpy array fot visualization purposes

def im_convert(tensor):

    '''Display tensor as an image'''

    image = tensor.to('cpu').clone().detach()

    image = image.numpy().squeeze()

    image = image.transpose(1, 2, 0)

    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))

    image = image.clip(0, 1)

    return image
plt.imshow(im_convert(content))
plt.imshow(im_convert(style))
vgg
def get_features(image, model, layers=None):

    """ Run an image forward through a model and get the features for 

        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)

    """

    

    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper

    ## Need the layers for the content and style representations of an image

    if layers is None:

        layers = {'0': 'conv1_1',

                  '5': 'conv2_1', 

                  '10': 'conv3_1', 

                  '19': 'conv4_1',

                  '21': 'conv4_2',  ## content representation

                  '28': 'conv5_1'}

        

    features = {}

    x = image

    # model._modules is a dictionary holding each module in the model

    for name, layer in model._modules.items():

        x = layer(x)

        if name in layers:

            features[layers[name]] = x

            

    return features
def gram_matrix(tensor):

    """ Calculate the Gram Matrix of a given tensor 

        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix

    """

    

    # get the batch_size, depth, height, and width of the Tensor

    _, d, h, w = tensor.size()

    

    # reshape so we're multiplying the features for each channel

    tensor = tensor.view(d, h * w)

    

    # calculate the gram matrix

    gram = torch.mm(tensor, tensor.t())

    

    return gram 
# get content and style features only one before forming target image

content_features = get_features(content, vgg)

style_features = get_features(style, vgg)
# Calculating the gram matrix for each layer of our style respresntation

style_grams = {layer : gram_matrix(style_features[layer]) for layer in style_features}
# create a third "target" image and prep it for change

# it is a good idea to start off with the target as a copy of our *content* image

# then iteratively change its style

target = content.clone().requires_grad_(True).to(device)
# Style weights for each of the style layers

# Weighting earlier layer more will result in Larger styke artifacts

style_weights = {'conv1_1': 1.,

                 'conv2_1': 0.75,

                 'conv3_1': 0.2,

                 'conv4_1': 0.2,

                 'conv5_1': 0.2}

content_weight = 1 # alpha

style_weight = 1e4 # large beta
# For displaying target image intermittenly

show_every = 1000



# iteration hyperparametes

optimizer = optim.Adam([target], lr=0.003)

steps = 3000 # decide how many iteration to update image (5000)



for ii in range (1, steps+1):

    

    # Get feature from target image

    target_features = get_features(target, vgg)



    # The content loss

    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

    

    # the style  loss

    # initialize style loss to zero

    style_loss = 0

    for layer in style_weights:

        target_feature = target_features[layer]

        target_gram = gram_matrix(target_feature)

        _, d, w, h = target_feature.shape

        

        # get stype representation 

        style_gram = style_grams[layer]

        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) **2)

        

        # Add the style loss

        style_loss += layer_style_loss / (d * h * w)

        

    # Calculate total loss

    total_loss = content_weight * content_loss + style_weight* style_loss





    

    # Updating target image

    optimizer.zero_grad()

    total_loss.backward()

    optimizer.step()

    

    # display intermediate images and print the loss

    if  ii % show_every == 0:

        print('Total loss: ', total_loss.item())

        plt.imshow(im_convert(target))

        plt.show()
# display content and final, target image

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(im_convert(content))

ax2.imshow(im_convert(target))