%matplotlib inline



from PIL import Image

import matplotlib.pyplot as plt

import numpy as np



import torch

import torch.optim as optim

from torchvision import transforms, models

vgg = models.vgg19(pretrained=True).features



for param in vgg.parameters():

  param.requires_grad_(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)
from urllib.request import urlopen



def load_image(img_path,max_size = 400,shape = None):

  image = Image.open(urlopen(img_path)).convert('RGB')

  if max(image.size) > max_size:

    size = max_size

  else:

    size = max(image.size)

  

  if shape is not None:

    size = shape



  in_transform = transforms.Compose([

                transforms.Resize(size),

                transforms.ToTensor(),

                transforms.Normalize((0.485,0.456,0.406),

                                      (0.299,0.224,0.225)

                                     )

  ])

  image = in_transform(image)[:3,:,:].unsqueeze(0)

  return image
style_path = "https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg"

content_path = "https://i.pinimg.com/736x/f7/b0/97/f7b09700344abe8af4beea276c83ca1f.jpg"



content = load_image(content_path).to(device)

style = load_image(style_path,shape=content.shape[-2:]).to(device)
def im_convert(tensor):

  image = tensor.to("cpu").clone().detach()

  image = image.numpy().squeeze()

  image = image.transpose(1,2,0)

  image =image * np.array((0.229,0.224,0.225)) + np.array((0.485,0.456,0.406))

  image = image.clip(0,1)

  return image
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20,10))

ax1.imshow(im_convert(content))

ax2.imshow(im_convert(style))
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
content_features = get_features(content,vgg)

style_features = get_features(style,vgg)



style_grams = {layer:gram_matrix(style_features[layer]) for layer in style_features }



target = content.clone().requires_grad_(True).to(device)
#weights



style_weights = {

    'conv1_1' : 1,

    'conv2_1' : 0.8,

    'conv3_1' : 0.5,

    'conv4_1' : 0.3,

    'conv5_1' : 0.1,

}

content_weight = 1

style_weight = 1e6

show_every = 100



optimizer = optim.Adam([target],lr = 0.003)

steps = 3000



for ii in range(0,steps +1):



  target_features = get_features(target,vgg)

  content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)



  style_loss = 0

  

  for layer in style_weights:



    target_feature = target_features[layer]

    target_gram = gram_matrix(target_feature)



    _,d,h,w = target_feature.shape

    style_gram = style_grams[layer]



    layer_style_loss = style_weights[layer] * torch.mean((target_gram -style_gram )**2)

    style_loss += layer_style_loss / (d*h*w)

  total_loss = content_weight *content_loss + style_weight*style_loss

  optimizer.zero_grad()

  total_loss.backward()

  optimizer.step()



  if ii % show_every == 0:

    print('Total loss:',total_loss.item())

    plt.imshow(im_convert(target))

    plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(im_convert(content))

ax2.imshow(im_convert(target))