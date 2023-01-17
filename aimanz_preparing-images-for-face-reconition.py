import numpy as np

from random import randint

from skimage.transform import resize, rotate,rescale

from skimage.feature import hog

import os

from skimage import data, io, util, exposure

from skimage.color import rgb2gray

import matplotlib.pyplot as plt

import math

import matplotlib.patches as ptchs

from skimage.exposure import equalize_adapthist
loadData = lambda dir : np.array([ util.img_as_float(e) \

                                  for e in io.imread_collection(dir)])
label = np.genfromtxt("../input/label.txt", delimiter=" ", dtype=int)

data = loadData("../input/*.jpg")
io.imshow(data[1])
data[1].shape
fig,ax = plt.subplots(1)

ax.imshow(data[1],cmap='gray')

for window in label[label[:,0] == 2]:

    rect = ptchs.Rectangle((window[2],window[1]),window[4],window[3],linewidth=2,edgecolor="red",facecolor='none')

    # plus le score est élevé plus le réctangle est rouge

    ax.add_patch(rect)

plt.show()
[imIdx, x, y, h, l] = label[0,:]



#[x-22:(x+h)+22, y-22:(y+l)+22]

H =90 

L =60

theta = 10

padding_left = math.ceil( (l/2)*math.cos(math.radians(theta)) + (h/2)*math.sin(math.radians(theta)))

padding_left = math.ceil( padding_left - l/2)





padding_top = math.ceil(( h/2)*math.cos(math.radians(theta)) + (l/2)* math.sin(math.radians(theta)))

padding_top = math.ceil( padding_top - h/2)



realI = max(0,x-padding_top)

realH = min(data[imIdx-1].shape[0],(x+h)+padding_top)



realJ = max(0,y-padding_left)

realL = min(data[imIdx-1].shape[1],(y+l)+padding_left)



im_padded = data[imIdx-1][realI:realH, realJ:realL]

im = data[imIdx-1][x:(x+h), y:(y+l)]





im_p_rot = rotate(im_padded,angle=theta,mode='reflect')

im_p_rot = im_p_rot[x - realI:(x+h) - realH, y- realJ:(y+l) -realL]



im_p_rot_sym = im_p_rot[:,::-1]





hog_vec, hog_vis = hog(resize(im,(H,L)), visualize=True, block_norm ='L2-Hys')



fig, ax = plt.subplots(2, 4, figsize=(12, 6),

                     subplot_kw=dict(xticks=[], yticks=[]))



ax[0,0].imshow(resize(im,(H,L)),cmap='gray')

ax[0,0].set_title('input image original')

ax[1,0].imshow(hog_vis)

ax[1,0].set_title('visualization of HOG')



ax[0,1].imshow(resize(im[:,::-1],(H,L)), cmap='gray')

ax[0,1].set_title('input image symetric')

ax[1,1].imshow(hog(resize(im[:,::-1],(H,L)), visualize=True,block_norm ='L2-Hys')[1])

ax[1,1].set_title('HOG of symetric')



ax[0,2].imshow(resize(im_p_rot,(H,L)),cmap='gray')

ax[0,2].set_title('input image rotated')

ax[1,2].imshow(hog(resize(im_p_rot,(H,L)), visualize=True,block_norm ='L2-Hys')[1])

ax[1,2].set_title('HOG of rotated')



ax[0,3].imshow(resize(im_p_rot_sym,(H,L)),cmap='gray')

ax[0,3].set_title('input image symetric of rotated')

ax[1,3].imshow(hog(resize(im_p_rot_sym,(H,L)), visualize=True,block_norm ='L2-Hys')[1])

ax[1,3].set_title('HOG of rotated symetric')