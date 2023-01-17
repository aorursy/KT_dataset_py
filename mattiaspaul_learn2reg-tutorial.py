import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))

plt.imshow(plt.imread('../input/learn2reg/concept_learn2reg.png'))

plt.axis('off')
# This cell imports all requires packages and provides namespace aliases

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import scipy.io

import numpy as np

import matplotlib.pyplot as plt

import os

print(os.listdir("../input/learn2reg"))

import sys

# insert at 1, 0 is the script path (or '' in REPL)

sys.path.insert(1, '../input/learn2reg')



from learn2reg_discrete import *

# Here, we import the image data and perform some preprocessing on the grayvalue images.

# In addition, to every slice, we provide organ segmentations that will be used for registration evaluation

# purposes (Dice scores) as well as for the weakly supervised training of the feature extraction 



# load TCIA data

imgs = torch.load('../input/learn2reg/tcia_prereg_2d_img.pth').float()/255

segs = torch.load('../input/learn2reg/tcia_prereg_2d_seg.pth').long()



# show an example patient slice

plt_pat = 17

plt.figure()

plt.subplot(121)

plt.imshow(imgs[plt_pat,:,:].cpu(),'gray')

plt.subplot(122)

plt.imshow(segs[plt_pat,:,:].cpu())



print('# labels:', segs.unique().size(0))
# Uncomment to use pretrained MIND-net to extract features

net = torch.load('../input/learn2reg/mindnet_cnn_pancreas.pth')



torch.manual_seed(10)



pat_indices = torch.cat((torch.arange(0,17),torch.arange(18,43)),0)



rnd_perm_idc = torch.randperm(pat_indices.size(0))

pat_indices = pat_indices[rnd_perm_idc]

train_set = pat_indices[:35]

test_set = torch.cat((pat_indices[35:],torch.LongTensor([17])),0)



# Now, we prepare our train & test dataset. 

test_set = torch.LongTensor([35,41,0,4,33,38,39,17])

train_set = torch.arange(43)

for idx in test_set:

    train_set = train_set[train_set != idx]



print('Test_Set:',test_set)

print('Train_Set:',train_set)
p_fix = train_set[9]

p_mov = train_set[15]





img_fixed = imgs[p_fix:p_fix+1,:,:].unsqueeze(1)#.to(crnt_dev)

img_moving = imgs[p_mov:p_mov+1,:,:].unsqueeze(1)#.to(crnt_dev)

seg_fixed = segs[p_fix:p_fix+1,:,:]

seg_moving = segs[p_mov:p_mov+1,:,:]

feat_fixed = net(img_fixed)

feat_moving = net(img_moving)



plt.figure(figsize=(12,8))

plt.subplot(231)

plt.imshow(img_fixed[0,0,:,:].cpu().data,'gray')

plt.subplot(232)

plt.imshow(feat_fixed[0,0,:,:].cpu().data)

plt.subplot(233)

plt.imshow(seg_fixed[0,:,:].cpu().data)

plt.subplot(234)

plt.imshow(img_moving[0,0,:,:].cpu().data,'gray')

plt.subplot(235)

plt.imshow(feat_moving[0,0,:,:].cpu().data)

plt.subplot(236)

plt.imshow(seg_moving[0,:,:].cpu().data)

plt.show()
displace_range = 9

disp_hw = (displace_range-1)//2



B,C,H,W = feat_fixed.size()

print(feat_fixed.size())

ssd_distance = correlation_layer(displace_range, feat_moving, feat_fixed)

print(ssd_distance.size())

soft_cost,xi,yi = meanfield(ssd_distance, img_fixed, displace_range, H,W)

print(soft_cost.size())





warp_and_evaluate(xi,yi, img_fixed, img_moving, seg_fixed, seg_moving , displace_range, H, W)

# The network defined here has the same architecture as the pretrained network above and we will train it from scratch 

# on the given image data.

net = torch.nn.Sequential(torch.nn.Conv2d(1,32,kernel_size=5,stride=2,padding=4,dilation=2),

                          torch.nn.BatchNorm2d(32),

                          torch.nn.PReLU(),

                          torch.nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,dilation=1),

                          torch.nn.BatchNorm2d(32),

                          torch.nn.PReLU(),

                          torch.nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1,dilation=1),

                          torch.nn.BatchNorm2d(64),

                          torch.nn.PReLU(),

                          torch.nn.Conv2d(64,24,kernel_size=1,stride=1,padding=0,dilation=1),

                          torch.nn.Sigmoid())

def my_correlation_layer(displace_range, feat_moving, feat_fixed):

    # TODO IMPLEMENT THE CORRELATION LAYER (or find the solution in learn2reg_discrete.py)

    

    # tensor dimensionalities in comments are for an arbitrary choice of

    # displace_range = 11 & feat sizes of [1,24,80,78];

    # they clearly depend on the actual choice and only serve as numerical examples here.

    

    disp_hw = (displace_range-1)//2

    # feat_mov: [1,24,80,78] -> 24 feature channels + spatial HW dims

    # feat_mov_unfold: [24,121,6240] -> mind chans, 11*11 = 121 displ steps, 6240 = 80*78 spatial positions

    feat_moving_unfold = F.unfold(feat_moving.transpose(1,0),(displace_range,displace_range),padding=disp_hw)

    B,C,H,W = feat_fixed.size()

    

    # feat_fixed: [24,1,6240] -> compute scalarproduct along feature dimension per broadcast + sum along 0

    # and reshape to [1,121,80,78]

    ssd_distance = ((feat_moving_unfold-feat_fixed.view(C,1,-1))**2).sum(0).view(1,displace_range**2,H,W)

    #reshape the 4D tensor back to spatial dimensions

    return ssd_distance#.detach()

ssd_distance = my_correlation_layer(displace_range, feat_moving, feat_fixed)

print(ssd_distance.size())
#net = torch.load('mindnet_cnn.pth')

#net.to(crnt_dev)

net.train()



optimizer = optim.Adam(list(net.parameters()),lr=0.00025)



nr_train_pairs = 50

grad_accum = 4



for pdx in range(nr_train_pairs):

    rnd_train_idx = torch.randperm(train_set.size(0))

    p_fix = train_set[rnd_train_idx[0]]

    p_mov = train_set[rnd_train_idx[1]]



    img_fixed = imgs[p_fix:p_fix+1,:,:].unsqueeze(1)#.to(crnt_dev)

    img_moving = imgs[p_mov:p_mov+1,:,:].unsqueeze(1)#.to(crnt_dev)

    feat_fixed = net(img_fixed)

    feat_moving = net(img_moving)



    seg_fixed = segs[p_fix:p_fix+1,:,:]

    seg_moving = segs[p_mov:p_mov+1,:,:]

    

    label_moving = F.one_hot(seg_moving,num_classes=9).permute(0,3,1,2).float()

    _,C1,Hf,Wf = label_moving.size()

    label_moving = F.interpolate(label_moving,size=(Hf//4,Wf//4),mode='bilinear')

    label_fixed = F.one_hot(seg_fixed,num_classes=9).permute(0,3,1,2).float()

    label_fixed = F.interpolate(label_fixed,size=(Hf//4,Wf//4),mode='bilinear')

    # generate the "unfolded" version of the moving encoding that will result in the shifted versions per channel

    # according to the corresponding discrete displacement pair

    label_moving_unfold = F.unfold(label_moving,(displace_range,displace_range),padding=disp_hw).view(1,9,displace_range**2,-1)

    



    #forward path: pass both images through the network so that the weights appear in the computation graph

    # and will be updated

    feat_fixed = net(img_fixed)

    feat_moving = net(img_moving)

    # compute the cost tensor using the correlation layer

    ssd_distance = my_correlation_layer(displace_range, feat_moving, feat_fixed)

    

    # compute the MIN-convolution & probabilistic output with the given function

    soft_cost,xi,yi = meanfield(ssd_distance, img_fixed, displace_range, H, W)

    # loss computation:

    # compute the weighted sum of the shifted moving label versions 

    label_warped = torch.sum(soft_cost.cpu().t().unsqueeze(0)*label_moving_unfold.squeeze(0),1)

    # compute the loss as sum of squared differences between the fixed label representation and the "warped" labels

    label_distance1 = torch.sum(torch.pow(label_fixed.view(8,-1)-label_warped.view(8,-1),2),0)

    loss = label_distance1.mean()

    # perform the backpropagation and weight updates

    loss.backward()

    

    if (pdx+1)%grad_accum == 0:

        # every grad_accum iterations : backpropagate the accumulated gradients

        optimizer.step()

        optimizer.zero_grad()



    if(pdx%(nr_train_pairs/4)==((nr_train_pairs/4)-1)):

        print(pdx,loss.item())



# Validate:

valid_pat_fix_idx = -1 # patient 17

valid_pat_mov_idx = 0



p_fix = test_set[valid_pat_fix_idx] # pat17

p_mov = test_set[valid_pat_mov_idx]





# 1) compute the feature representations

net = net.eval()

img_fixed = imgs[p_fix:p_fix+1,:,:].unsqueeze(1)#.to(crnt_dev)

img_moving = imgs[p_mov:p_mov+1,:,:].unsqueeze(1)#.to(crnt_dev)

feat_fixed = net(img_fixed)

feat_moving = net(img_moving)



seg_fixed = segs[p_fix:p_fix+1,:,:]

seg_moving = segs[p_mov:p_mov+1,:,:]



# 2) perform the SSD cost calculation based on the correlation layer 

ssd_distance = correlation_layer(displace_range, feat_moving, feat_fixed)





soft_cost,xi,yi = meanfield(ssd_distance, img_fixed, displace_range, H, W)



warp_and_evaluate(xi,yi, img_fixed, img_moving, seg_fixed, seg_moving , displace_range, H, W)
plt.figure(figsize=(20,8))

plt.imshow(plt.imread('../input/learn2reg/cnn_matmul.png'))

plt.axis('off')
plt.figure(figsize=(20,8))

plt.imshow(plt.imread('../input/learn2reg/unfold_tensors.png'))

plt.axis('off')
plt.figure(figsize=(20,8))

plt.imshow(plt.imread('../input/learn2reg/discrete_weights.png'))

plt.axis('off')