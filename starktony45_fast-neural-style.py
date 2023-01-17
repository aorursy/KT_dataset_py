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
import torch
from torchvision import datasets,transforms
# download helper code
!wget https://raw.githubusercontent.com/iamRusty/fast-neural-style-pytorch/master/transformer.py
!wget https://raw.githubusercontent.com/iamRusty/fast-neural-style-pytorch/master/utils.py
!wget https://raw.githubusercontent.com/iamRusty/fast-neural-style-pytorch/master/vgg.py
    
#download model
!wget https://web.eecs.umich.edu/~justincj/models/vgg16-00b39a1b.pth
!wget https://raw.githubusercontent.com/iamRusty/fast-neural-style-pytorch/master/images/mosaic.jpg
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import random
import time
import utils
import transformer
import vgg
#global settings
train_image_size = 256
dataset_path = "/kaggle/input/image-dataset"
num_epochs = 1
style_image_path = "/kaggle/working/mosaic.jpg"
batch_size = 4
content_weight = 17
style_weight = 50
tv_weight = 1e-6
adam_lr = 0.001
save_model_path = '/kaggle/working/'
save_image_path = '/kaggle/working/'
save_model_every = 500
seed = 9






def train():
    #seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([transforms.Resize(train_image_size),
                                   transforms.CenterCrop(train_image_size),
                                   transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x.mul(255))
                                   ])
    
    train_dataset = datasets.ImageFolder(dataset_path,transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size =4,shuffle =True)
    
    #Load networks
    TransformerNetwork = transformer.TransformerNetwork().to(device)
    VGG = vgg.VGG16("/kaggle/working/vgg16-00b39a1b.pth").to(device)
    
    #Get style features
    imagenet_neg_mean = torch.tensor([-103.939, -116.779, -123.68], dtype = torch.float32).reshape(1,3,1,1).to(device)
    
    imagenet_mean = torch.tensor([103.939, 116.779, 123.68],dtype =torch.float32).reshape(1,3,1,1).to(device)
    
    style_image = utils.load_image(style_image_path)
    style_tensor = utils.itot(style_image).to(device)
    style_tensor = style_tensor.add(imagenet_neg_mean)
    b,c,h,w = style_tensor.shape
    style_features = VGG(style_tensor.expand(batch_size,c,h,w))
    
    style_gram = {}
    for key,values in style_features.items():
        style_gram[key]=utils.gram(values)
        
    #optimizer settings
    
    
    optimizer = optim.Adam(TransformerNetwork.parameters(),lr =adam_lr)
    
    #loss_trackers 
    content_loss_history = []
    style_loss_history = []
    total_loss_history =[]
    batch_content_loss_sum = 0
    batch_style_loss_sum = 0
    batch_total_loss_sum = 0
    
    #optimization/training_loop
    batch_count =1
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        print("========Epoch{}/{}=====".format(epoch,num_epochs+1))
        for batch_id,(content_batch,_) in enumerate(train_loader):
            cur_batch_size = content_batch.shape[0]
            
            optimizer.zero_grad()
            
            #Generate Iimages
            content_batch = content_batch[:,[2,1,0]].to(device)
            generated_batch = TransformerNetwork(content_batch)
            content_features = VGG(content_batch.add(imagenet_neg_mean))
            generated_features = VGG(generated_batch.add(imagenet_neg_mean))
            
            #Content loss
            MSELoss = nn.MSELoss().to(device)
            content_loss = content_weight * MSELoss(content_features['relu2_2'], generated_features['relu2_2'])
            batch_content_loss_sum +=content_loss
            
            #Style loss
            style_loss = 0
            for key,value in generated_features.items():
                s_loss = MSELoss(utils.gram(value), style_gram[key][:cur_batch_size])
                style_loss += s_loss
            style_loss*= style_weight
            batch_style_loss_sum += style_loss
            
            #total loss
            total_loss = content_loss +style_loss
            batch_total_loss_sum +=total_loss.item()
            
            #Back propogation
            total_loss.backward()
            optimizer.step()
            
            if (((batch_count-1)%save_model_every == 0) or (batch_count==num_epochs*len(train_loader))):
                # Print Losses
                print("========Iteration {}/{}========".format(batch_count, num_epochs*len(train_loader)))
                print("\tContent Loss:\t{:.2f}".format(batch_content_loss_sum/batch_count))
                print("\tStyle Loss:\t{:.2f}".format(batch_style_loss_sum/batch_count))
                print("\tTotal Loss:\t{:.2f}".format(batch_total_loss_sum/batch_count))
                print("Time elapsed:\t{} seconds".format(time.time()-start_time))

                # Save Model
                checkpoint_path = save_model_path + "checkpoint_" + str(batch_count-1) + ".pth"
                torch.save(TransformerNetwork.state_dict(), checkpoint_path)
                print("Saved TransformerNetwork checkpoint file at {}".format(checkpoint_path))

                # Save sample generated image
                sample_tensor = generated_batch[0].clone().detach().unsqueeze(dim=0)
                sample_image = utils.ttoi(sample_tensor.clone().detach())
                sample_image_path = save_image_path + "sample0_" + str(batch_count-1) + ".png"
                utils.saveimg(sample_image, sample_image_path)
                utils.show(sample_image)
                print("Saved sample tranformed image at {}".format(sample_image_path))

                # Save loss histories
                content_loss_history.append(batch_total_loss_sum/batch_count)
                style_loss_history.append(batch_style_loss_sum/batch_count)
                total_loss_history.append(batch_total_loss_sum/batch_count)

            # Iterate Batch Counter
            batch_count+=1

    stop_time = time.time()
    # Print loss histories
    print("Done Training the Transformer Network!")
    print("Training Time: {} seconds".format(stop_time-start_time))
    print("========Content Loss========")
    print(content_loss_history) 
    print("========Style Loss========")
    print(style_loss_history) 
    print("========Total Loss========")
    print(total_loss_history) 

    # Save TransformerNetwork weights
    TransformerNetwork.eval()
    TransformerNetwork.cpu()
    final_path = save_model_path + "transformer_weight.pth"
    print("Saving TransformerNetwork weights at {}".format(final_path))
    torch.save(TransformerNetwork.state_dict(), final_path)
    print("Done saving final model")

train()
            
            
    
    
    
    
    
    
    
