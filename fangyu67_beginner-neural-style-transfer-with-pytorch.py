import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#import matplotlib.image as mpimg

%matplotlib inline



import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

#from torch.utils.data import Dataset

from torchvision.models import vgg19

import torchvision.transforms as transforms

#from collections import OrderedDict

%pylab inline

print('Pytorch version: {}'.format(torch.__version__))





from PIL import Image

import time

import os

print(os.listdir("../input"))



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Computational device: {}'.format(device))
StylePath = '/kaggle/input/styles/'

ContentPath = '/kaggle/input/content/'



name_imgC = 'WechatIMG1.jpeg'

name_imgS = 'picasso1.jpg'



cont_image_path = ContentPath + name_imgC

style_image_path = StylePath + name_imgS



imageC = Image.open(cont_image_path)

imageS = Image.open(style_image_path)
fig, ax = plt.subplots(1,2, figsize=(18, 12))

ax[0].set_title('Content image', fontsize="20")

ax[0].imshow(imageC.resize((512,512)))  

ax[1].set_title('Style image', fontsize="20")

ax[1].imshow(imageS.resize((512,512)))
imsize = 512



prep = transforms.Compose([transforms.Resize((imsize,imsize)),

                           transforms.ToTensor(),

                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), 

                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean

                                                std=[1,1,1]),

                           transforms.Lambda(lambda x: x.mul_(255)),

                          ])





postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1./255)),

                           transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], #add imagenet mean

                                                std=[1,1,1]),

                           transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to RGB

                           ])





postpb = transforms.Compose([transforms.ToPILImage()])



# Preprocessing: convert image to array

def prep_img(img_path):

    image = Image.open(img_path)

    image = Variable(prep(image))

    # network's input need at least a batch size 1

    image = image.unsqueeze(0)

    return image.to(device,torch.float)









# Postprocessing: convert array to image

def postp(tensor): 

    t = postpa(tensor)

    # to clip results in the range [0,1]

    t[t>1] = 1    

    t[t<0] = 0

    img = postpb(t)

    return img
# input style image and content inmage

style_img = prep_img(style_image_path)

content_img = prep_img(cont_image_path)

#assert style_img.size() == content_img.size(),"import style and content images are not in the same size"



# load model in eval mode (model uses BN or Dropout)

vgg = vgg19(pretrained=True).features.to(device).eval()



# set requires_grad as false, as a result no backprop of the gradients

for param in vgg.parameters():

    param.requires_grad = False

#    print(param.requires_grad)





# initialize the output image as same as the content image or a random noise. The image need to be modified. 

opt_img = Variable(content_img.data.clone(),requires_grad=True)



#input_img = torch.randn(content_img.data.size(), device=device)

#opt_img = Variable(input_img,requires_grad=True)
#style_img.shape

#content_img.shape

#opt_img

vgg
# choose layers for style

style_layers = [1,6,11,20,26,35]



# one layer for content

content_layers = [29]
# use hook to extract activations during forward prop

class LayerActivations():

    features=[]

    

    def __init__(self,model,layer_nums):

        

        self.hooks = []

# register activation after forword at eatch layer 

        for layer_num in layer_nums:

            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))

#     

    def hook_fn(self,module,input,output):

        self.features.append(output)



    

    def remove(self):

        for hook in self.hooks:

            hook.remove()
def extract_layers(layers,img,model=None):

    la = LayerActivations(model,layers)

    #Clearing the cache 

    la.features = []

    # forward prop img and hook registes automatically activations

    out = model(img)

    # remove hook but features are already extracted.

    la.remove()

    return la.features
class ContentLoss(nn.Module):    

    

    def forward(self,inputs,targets):

        assert inputs.size() == targets.size(),"need the same size"

        b,c,h,w = inputs.size()

        loss = nn.MSELoss()(inputs, targets)

        loss.div_(4*c*h*w)

        return (loss)
class GramMatrix(nn.Module):

    

    def forward(self,input):

# batch, channel, height, width        

        b,c,h,w = input.size()

        features = input.view(b,c,h*w)

# batch matrix product (b*n*m)        

        gram_matrix =  torch.bmm(features,features.transpose(1,2))

        return gram_matrix
class StyleLoss(nn.Module):

    def forward(self,inputs,targets):

        assert inputs.size() == targets.size(),"need the same size"

        b,c,h,w = inputs.size()

        loss = F.mse_loss(GramMatrix()(inputs), GramMatrix()(targets))

        loss.div_(4*(c*h*w)**2)

        return (loss)
a_c = extract_layers(content_layers,content_img,model=vgg)

a_c = [t.detach() for t in a_c]



a_s = extract_layers(style_layers,style_img,model=vgg)

a_s = [t.detach() for t in a_s]



activations = a_s + a_c 

#activations
loss_fns = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)
# weight of layers (alpha and beta)

style_weights = [100000 for n in range(len(style_layers))]

content_weights = [1]

weights = style_weights + content_weights
max_iter = 500

show_iter = 100



# parameters to optimize (tensors or dicts)

optimizer = optim.LBFGS([opt_img]);

n_iter=[0]



while n_iter[0] <= max_iter:

    

    # evaluate the model and return the loss for optimizer

    def closure():

        

        # clean cach

        optimizer.zero_grad()

        

        # extract acivations of the output image

        out_sty = extract_layers(style_layers,opt_img,model=vgg)

        out_cnt = extract_layers(content_layers,opt_img,model=vgg)

        out =  out_sty + out_cnt

        

        # compute losses

        layer_losses = [weights[a] * loss_fns[a](A, activations[a]) for a,A in enumerate(out)]

        #print(layer_losses[0])

        

        # .backward apply to a scaler

        loss = sum(layer_losses)

        

        # compute gradients

        loss.backward()

        n_iter[0]+=1

        

        if n_iter[0]%show_iter == (show_iter-1):

            print('Iteration: %d, loss: %f'%(n_iter[0]+1, loss.item()))



        return loss

    # parameters update

    optimizer.step(closure)

    
#display result

out_img_hr = postp(opt_img.data[0].cpu().squeeze())



imshow(out_img_hr)

gcf().set_size_inches(10,10)