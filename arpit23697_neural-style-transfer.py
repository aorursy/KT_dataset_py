import numpy as np

import pandas as pd

import time

import matplotlib.pyplot as plt



import torch

from torch.optim import lr_scheduler

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, datasets, models

from torchvision.datasets import ImageFolder

import torch.optim as optim

import torch.nn as nn

import torch.nn.functional as F



from PIL import Image
vgg = models.vgg19(pretrained = True)
vgg
imsize = 512

is_cuda = torch.cuda.is_available()
#converting the image making it suitable for training with the vgg model

prep = transforms.Compose([transforms.Resize(imsize) ,

                           transforms.ToTensor(),

                           transforms.Lambda(lambda x : x[torch.LongTensor([2,1,0])]) , #turn it into BGR 

                           transforms.Normalize(mean = [0.40760392 , 0.45795686 , 0.48501961],

                                                std = [1,1,1]),

                           transforms.Lambda(lambda x : x.mul_(255)),

                          ])
#converting the generated image back to a format which we can visualize

postpa = transforms.Compose([transforms.Lambda(lambda x : x.mul_(1./255)),

                            transforms.Normalize(mean = [-0.4076032 , -0.45795686 , -0.48501961] , 

                                                std = [1,1,1]),

                            transforms.Lambda(lambda x : x[torch.LongTensor([2,1,0])]) ,  #turn it back into RGB

                            ])



postpb = transforms.Compose([transforms.ToPILImage()])

#to ensure that the generate image do not cross the permissible range

def postp(tensor):

    t = postpa(tensor)

    t[t > 1] = 1

    t[t < 0] = 0

    img = postpb(t)

    return img
def image_loader(image_name):

    image = Image.open(image_name)        #takes the image and preprocess it for training

    image = Variable(prep(image))

    image = image.unsqueeze(0)          #fake batch dimension required to fit network's input dimension

    return image
def show_image(image , title):

    plt.figure(figsize=(12 , 8))

    plt.imshow(image)

    plt.xticks([])

    plt.yticks([])

    plt.title(title)
import os

print(os.listdir("../input"))
content_img_path = "../input/images/Content.jpg"

style_img_path = "../input/images/Style Image.jpg"
content = Image.open(content_img_path)

style = Image.open(style_img_path)
show_image(content , "Content image")
show_image(style , "Style Image")
style_img = image_loader(style_img_path)

content_img = image_loader(content_img_path)
# show_image(style_img[0][0], "Loaded style image B Channel")
# show_image(content_img[0][2].cpu() , "Loaded content image B channel")
content_img , style_img = content_img.cuda(), style_img.cuda()
opt_img = Variable(content_img.data.clone().cuda() , requires_grad= True)
vgg_features = vgg.features.cuda()
#Since we are not changing the weights of the vgg model so change requires_grad = False

for param in vgg_features.parameters():

    param.requires_grad = False
class GramMatrix(nn.Module):

    def forward(self, input):

        b,c,h,w = input.size()

        features = input.view(b,c,h*w)      #So dimension is b * c * (h*w) 

                                            #Gram matrix contains the correlation between different channels of the image

        gram_matrix = torch.bmm(features , features.transpose(1,2))

        gram_matrix.div_(h * w)          #normalizing the image

        return gram_matrix

        
#implemented as the nn.Layer

class styleLoss(nn.Module):

    def forward(self, inputs, targets):         #target is the style values of the style image

        out = nn.MSELoss()(GramMatrix()(inputs) , targets ) #pass the input through the image, gets its loss and compute the mse loss

        return out

    

    

    #GramMatrix()    - creates the class

    #pass the input through that model and get the gram matrix

    

    #Create nn.MSELoss and compute the loss for the style

    
class LayerActivations ():

    features = []

    def __init__(self , model , layer_nums):

        self.hooks = []

        for layer_num in layer_nums:

            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))

            

    def hook_fn (self , module , input , output):

        self.features.append(output)

        

    def remove(self):

        for hook in self.hooks:

            hook.remove();

        
def extract_layers(layers , img , model = None):

    la = LayerActivations(model , layers)

    

    #clearing the cache

    la.features = []

    out = model(img)

    la.remove()

    return la.features

    
style_layers = [1,6,11,20,25]

content_layers = [21]

loss_layers = style_layers + content_layers
style_weights = [1000/n**2 for n in [64 , 128 , 256 , 512 , 512]]

content_weights = [1]

weights = style_weights + content_weights
content_targets = extract_layers(content_layers , content_img , model = vgg_features)

style_targets = extract_layers(style_layers , style_img , model = vgg_features)
#detach the targets since we are not changing the content and the style image

content_targets = [t.detach() for t in content_targets]

style_targets = [GramMatrix()(t).detach() for t in style_targets]
#add the targets together

targets = style_targets + content_targets
loss_fns = [styleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
optimizer = optim.LBFGS([opt_img])
max_iter = 500

itr = [0]

while itr[0] <= max_iter:

    

    def closure():

        optimizer.zero_grad()

        out = extract_layers(loss_layers , opt_img , model = vgg_features)

        

        layer_losses = [weights[a] * loss_fns[a](A , targets[a]) for a,A in enumerate(out)]

        

        loss = sum(layer_losses)

        loss.backward()

        itr[0] += 1

        if (itr[0] % 50 == 0):

            print("Iteration :" , itr[0] , " loss : " , loss.item())

        

        return loss

    

    optimizer.step(closure)

        
out_img = postp(opt_img.data[0].cpu().squeeze())

show_image(out_img , "Output image")