# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load the model and see its architecture



import torch

import torch.nn as nn

from torch.autograd import Variable

from torchvision import models

from torchvision import transforms, utils



import numpy as np

import scipy.misc

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image



model = models.vgg16(pretrained=True)

print(model.features)
# We will load all the module details in a list



modules = list(model.features.modules())

modules = modules[1:]

print(modules,"\n\n")

print("third module = ", modules[2])
# Load and preprocess an image to pass as input to the network



def normalize(image):

    normalize = transforms.Normalize(

    mean=[0.485, 0.456, 0.406],

    std=[0.229, 0.224, 0.225]

    )

    preprocess = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    normalize

    ])

    image = Variable(preprocess(image).unsqueeze(0))

    return image



img_raw = Image.open("/kaggle/input/heart-image/heart.jpg")

plt.imshow(img_raw)

plt.title("Image loaded successfully")



img = normalize(img_raw)
def visualize_weights(image, layer):

    weight_used = []

    

    ## Gather all Convolution layers and append their corresponding filters in a list

    for w in model.features.children():

        if isinstance(w, torch.nn.modules.conv.Conv2d):

            weight_used.append(w.weight.data)



    print("(#filters, i/p depth, size of filter) === ",weight_used[layer].shape)

    print("No. of filters: ", weight_used[layer].shape[0])

    filters = []

    for i in range(weight_used[layer].shape[0]):

        filters.append(weight_used[layer][i,:,:,:].sum(dim=0))    ##summing across input depth(3 in the first layer)

        filters[i].div(weight_used[layer].shape[1])

        

    fig = plt.figure()

    plt.rcParams["figure.figsize"] = (10, 10)

    for i in range(int(np.sqrt(weight_used[layer].shape[0])) * int(np.sqrt(weight_used[layer].shape[0]))):

        a = fig.add_subplot(np.sqrt(weight_used[layer].shape[0]),np.sqrt(weight_used[layer].shape[0]),i+1)

        imgplot = plt.imshow(filters[i])

        plt.axis('off')



visualize_weights(img, 1)
# Visualizing the image as it passes through the network



def to_grayscale(image):

    image = torch.sum(image, dim=0)

    image = torch.div(image, image.shape[0])

    return image



def layer_outputs(image):

    outputs = []

    names = []

    

    ## feed forward the image through the network and store the outputs

    for layer in modules:

        image = layer(image) 

        outputs.append(image)

        names.append(str(layer))

    

    ## for visualization purposes, convert the output into a 2D image by averaging across the filters.

    output_im = []

    for i in outputs:

        i = i.squeeze(0)

        temp = to_grayscale(i)  ## convert say 64x112x112 to 112x112

        output_im.append(temp.data.numpy())

        

    fig = plt.figure()

    plt.rcParams["figure.figsize"] = (30, 40)





    for i in range(len(output_im)):

        a = fig.add_subplot(8,4,i+1)

        imgplot = plt.imshow(output_im[i])

        plt.axis('off')

        a.set_title(str(i+1)+". "+names[i].partition('(')[0], fontsize=15)



#     ##save the resulting visualization

#     plt.savefig('layer_outputs.jpg', bbox_inches='tight')



layer_outputs(img)
# Visualizing output of each filter at a given layer



def filter_outputs(image, layer_to_visualize, num_filters=64):

    if layer_to_visualize < 0:

        layer_to_visualize += 31

    output = None

    name = None

    #image at each layer

    ## get outputs corresponding to the mentioned layer

    for count, layer in enumerate(modules):

        image = layer(image)

        if count == layer_to_visualize: 

            output = image

            name = str(layer)

    

    filters = []

    output = output.data.squeeze()



    ## if num_filters==-1, visualize all the filters

    num_filters = min(num_filters, output.shape[0])

    if num_filters==-1:

        num_filters = output.shape[0]



    for i in range(num_filters):

        filters.append(output[i,:,:])

        

    fig = plt.figure()

    plt.rcParams["figure.figsize"] = (10, 10)



    for i in range(int(np.sqrt(len(filters))) * int(np.sqrt(len(filters)))):

        fig.add_subplot(np.sqrt(len(filters)), np.sqrt(len(filters)),i+1)

        imgplot = plt.imshow(filters[i])

        plt.axis('off')



## if num_filters==-1, visualize all the filters

filter_outputs(img,0,16)    #visualize the outputs of first 16 filters of the 1st layer
# Understanding Deep Image Representations by Inverting Them [Mahendran, Vedaldi]



# Like Zeiler and Fergus, their method starts from a specific input image. They record the network’s representation of that specific image and then reconstruct an image that produces a similar code. Thus, their method provides insight into what the activation of a whole layer represent, not what an individual neuron represents.

# They show what each neuron “wants to see”, and thus what each neuron has learned to look for.

# To visualize the function of a specific unit in a neural network, we  synthesize  inputs that cause that unit to have high activation. To synthesize such a “preferred input example”, we start with a random image, meaning we randomly choose a color for each pixel. The image will initially look like colored TV static.





random_noise_img = Variable(1e-1 * torch.randn(1, 3, 224, 224), requires_grad=True)
# Now we take an image  X  whose representation  X0  at some layer  ‘‘target_layer"  we want to learn. Our aim is to reconstruct the noise image to get this representation  X0 . The principle behind this is that the noise image will be so reconstructed such that it will represent what the particular layer for which it is trained against wants to see.





def get_output_at_nth_layer(inp, layer):

    for i in range(layer):

        inp = modules[i](inp)

    return inp[0]



## dont forget that the system is 0 indexed

target_layer = 18    ## which is this layer Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

inp_img = normalize(Image.open("/kaggle/input/heart-image/heart.jpg"))

inp_img_representation = get_output_at_nth_layer(inp_img, target_layer)