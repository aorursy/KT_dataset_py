import numpy as np 

import pandas as pd 

import math

import zipfile

import scipy.optimize as op

import matplotlib.pyplot as plt

import cv2 as cv2

import tensorflow as tf
#Read image in Grayscale

im = cv2.imread('../input/SampleImage.JPG',cv2.IMREAD_GRAYSCALE)

im=cv2.resize(im, (100,100))

plt.axis('off')

plt.imshow(im)

plt.show()
print('Image Size=',im.shape)
#Filters

vfilter=np.array([[-1.,0.,1.],[-1.,0.,1.],[-1.,0.,1.]])

hfilter=np.array([[-1.,-1.,-1.],[0.,0.,0.],[1.,1.,1.]])

f=len(vfilter)



#Padding

pad=1

padImg=np.pad(im, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))



#Stride

stride=1



#Out image size calculations

n_H_prev,n_W_prev=im.shape

n_H = int((n_H_prev - f + 2 * pad) / stride + 1)     

n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

outImg=np.zeros((n_W,n_H))





print('out Image Size=',outImg.shape)
filter=hfilter

for i in range(n_H):

    for j in range(n_W):

        outImg[i,j]= np.sum(np.multiply(padImg[i*stride:i*stride+f,j*stride:j*stride+f],filter))

print('Output Image Size=',outImg.shape)

plt.subplot('121')

plt.imshow(outImg)

plt.title('Horizotal Edges')



filter=vfilter

for i in range(n_H):

    for j in range(n_W):

        outImg[i,j]= np.sum(np.multiply(padImg[i*stride:i*stride+f,j*stride:j*stride+f],filter))

print('Output Image Size=',outImg.shape)

plt.subplot('122')

plt.imshow(outImg)

plt.title('Vertical Edges')



plt.show()
#Read image in Grayscale

clrImg = cv2.imread('../input/SampleImage.JPG')

clrImg=cv2.resize(clrImg, (100,100))

plt.axis('off')

plt.imshow(clrImg)

plt.show()
X=np.array([clrImg])     # X=np.array([clrImg1,clrImg2,clrImg3,.....so on])

X.shape
def zero_pad(X, pad):

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))    

    return X_pad
m=1



#Filters

hfilter=np.array([[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]],

                  [[  0.,0., 0.],[ 0., 0., 0.],[ 0., 0., 0.]],

                  [[ 1., 1., 1.],[ 1., 1., 1.],[ 1., 1., 1.]]])



vfilter=np.array([[[-1., -1., -1.],[  0.,   0.,   0.],[ 1.,  1.,  1.]],

                  [[-1., -1., -1.],[  0.,   0.,   0.],[ 1.,  1.,  1.]],

                  [[-1., -1., -1.],[  0.,   0.,   0.],[ 1.,  1.,  1.]]])



print(vfilter.shape)

f=len(vfilter)



#Padding

pad=2

X_Pad= zero_pad(X, pad)



#Stride

stride=1



#Out image size calculations

m,n_H_prev,n_W_prev,n_C_prev=X.shape

n_H = int((n_H_prev - f + 2 * pad) / stride + 1)     

n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

n_C =1

                  

X_Out=np.zeros((m,n_W,n_H,n_C))

print('out Image Size=',X_Out.shape)


filter=vfilter

X_Out=np.zeros((m,n_W,n_H,n_C))

for i in range(m):                         # loop over the training examples

    for h in range(n_H):                 # loop on the vertical axis of the output volume

        for w in range(n_W):                 # loop on the horizontal axis of the output volume

            for c in range (n_C):                # loop over the channels of the output volume 

                X_Slice=X_Pad[i,h*stride:h*stride+f,w*stride:w*stride+f,:]

                X_Conv=np.multiply(X_Slice,filter)

                X_Out[i,h,w,c]= np.sum(X_Conv)

                



print('Output Image Size=',X_Out.shape)

plt.subplot('122')

plt.imshow(X_Out[0,:,:,0])

plt.title('Vertical Edges')



filter=hfilter

X_Out=np.zeros((m,n_W,n_H,n_C))

for i in range(m):                         # loop over the training examples

    for h in range(n_H):                 # loop on the vertical axis of the output volume

        for w in range(n_W):                 # loop on the horizontal axis of the output volume

            for c in range (n_C):                # loop over the channels of the output volume

                X_Slice=X_Pad[i,h*stride:h*stride+f,w*stride:w*stride+f,:]

                X_Conv=np.multiply(X_Slice,filter)

                X_Out[i,h,w,c]= np.sum(X_Conv)

                

print('Output Image Size=',X_Out.shape)

plt.subplot('121')

plt.imshow(X_Out[0,:,:,0])

plt.title('Horizotal Edges')

plt.show()


f=10      #Filter size for pooling

stride=3



m,n_H_prev,n_W_prev,n_C_prev=X.shape

n_H = int(1 + (n_H_prev - f) / stride)

n_W = int(1 + (n_W_prev - f) / stride)

n_C = n_C_prev

                  

X_Max=np.zeros((m,n_W,n_H,n_C))

X_Avg=np.zeros((m,n_W,n_H,n_C))

print('out Image Size=',X_Max.shape)



for i in range(m):                         # loop over the training examples

    for h in range(n_H):                 # loop on the vertical axis of the output volume

        for w in range(n_W):                 # loop on the horizontal axis of the output volume

            for c in range (n_C):                # loop over the channels of the output volume

                X_Slice=X[i,h*stride:h*stride+f,w*stride:w*stride+f,c]

                X_Max[i,h,w,c]= np.max(X_Slice)

                X_Avg[i,h,w,c]= np.mean(X_Slice)

                



print('Output Image Size=',X_Max.shape)

plt.subplot('121')

plt.imshow((X_Max[0,:,:,:]).astype(int))

plt.title('Max Pooling')



plt.subplot('122')

plt.imshow((X_Avg[0,:,:,:]).astype(int))

plt.title('Avg Pooling')

plt.show()

                

                


#Convolve Step







#Filters or Weights

Weights=np.array([[[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]],

                  [[  0.,0., 0.],[ 0., 0., 0.],[ 0., 0., 0.]],

                  [[ 1., 1., 1.],[ 1., 1., 1.],[ 1., 1., 1.]]],



                [[[-1., -1., -1.],[  0.,   0.,   0.],[ 1.,  1.,  1.]],

                  [[-1., -1., -1.],[  0.,   0.,   0.],[ 1.,  1.,  1.]],

                  [[-1., -1., -1.],[  0.,   0.,   0.],[ 1.,  1.,  1.]]]])



f=len(Weights[0,:,:,:])



#Padding

pad=1

X_Pad= zero_pad(X, pad)



#Stride

stride=1



#Out image size calculations

m,n_H_prev,n_W_prev,n_C_prev=X.shape

n_H = int((n_H_prev - f + 2 * pad) / stride + 1)     

n_W = int((n_W_prev - f + 2 * pad) / stride + 1)

n_C =2

                  

X_Out=np.zeros((m,n_W,n_H,n_C))





for i in range(m):                         # loop over the training examples

    for h in range(n_H):                 # loop on the vertical axis of the output volume

        for w in range(n_W):                 # loop on the horizontal axis of the output volume

            for c in range (n_C):                # loop over the channels of the output volume 

                X_Slice=X_Pad[i,h*stride:h*stride+f,w*stride:w*stride+f,:]

                X_Conv=np.multiply(X_Slice,Weights[c,:,:,:])

                X_Conv=X_Conv+np.ones(X_Conv.shape)   #Add Bias Vector  W * X +b

                X_Out[i,h,w,c]= np.sum(X_Conv)  





                

                

#Relu Activation Step                   

X_Out = np.maximum(0,X_Out)     



#Sigmoid Action Step

#X_Out = 1/(1+np.exp(-X_Out))         

     

    

#Max Pooling Step

f=10      #Filter size for pooling

stride=3



m,n_H_prev,n_W_prev,n_C_prev=X_Out.shape

n_H = int(1 + (n_H_prev - f) / stride)

n_W = int(1 + (n_W_prev - f) / stride)

n_C = n_C_prev

X_Out=np.zeros((m,n_W,n_H,n_C))

for i in range(m):                         # loop over the training examples

    for h in range(n_H):                 # loop on the vertical axis of the output volume

        for w in range(n_W):                 # loop on the horizontal axis of the output volume

            for c in range (n_C):                # loop over the channels of the output volume

                X_Slice=X[i,h*stride:h*stride+f,w*stride:w*stride+f,c]

                X_Out[i,h,w,c]= np.max(X_Slice)     

                



print('Output Image Size=',X_Out.shape)

plt.subplot('121')

plt.imshow((X_Out[0,:,:,0]).astype(int))

plt.title('Convovle 1 Step')

                