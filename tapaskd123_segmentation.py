import os

import cv2

import seaborn as sns

import numpy as np 

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers

from tensorflow.keras.layers import add

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import classification_report, confusion_matrix



np.random.seed(777)

tf.random.set_seed(777)
BATCH_SIZE = 32

IMG_HEIGHT = 240

IMG_WIDTH = 240

ALPHA = 2e-4


def psnr(inp,out):

    den=np.sum((inp-out)**2)/(240**2)

    num=np.max(inp)

    print(den)

    return (np.log10((num**2)/den))*10
labels = ['PNEUMONIA', 'NORMAL']

def get_data(data_dir):

    data = [] 

    for label in labels: 

        path = os.path.join(data_dir, label)

        class_num = labels.index(label)

        for en,img in enumerate(os.listdir(path)):

            if en==0:

                print(img)

            try:

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) 

                resized_arr = cv2.resize(img_arr, (IMG_WIDTH, IMG_HEIGHT))

                data.append([resized_arr, class_num])

            except Exception as e:

                pass



    return np.array(data)
train = get_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/train')

test = get_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')

val = get_data('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')
import numpy as np

def pad1(a):

    b=np.ones((a.shape[0]+2,a.shape[1]+2))

    b[1:a.shape[0]+1,1:a.shape[1]+1]=a

    b[1:4,[0,-1]]=b[1:4,[1,-2]]

    b[[0,-1],:]=b[[1,-2],:]

    return b


def conv(inp,ker):

    inp=pad1(inp)

    v_k,h_k=ker.shape

    v_i,h_i=inp.shape

    out=np.zeros((1+v_i-v_k,1+h_i-h_k))

    for i in range(v_i):

        for j in range(h_i):

            if (i+v_k<=v_i) and (j+h_k<=h_i):

                out[i,j]=np.sum(inp[i:i+v_k,j:j+h_k]*ker)

    return out
#speck, slat and pepper
plt.imshow(train[0][0],cmap='gray')

np.save('original.npy',train[0][0])
#mean filter

filter=np.ones((3,3))

filter/=np.sum(filter)

out=conv(train[0][0],filter)

plt.imshow(out,cmap='gray')

np.save('mean.npy',out)
psnr(train[0][0],out)
#gaussian filter

filter=np.array([[1.0,2.0,1.0], [2.0,4.0,2.0], [1.0,2.0,1.0]])

filter/=np.sum(filter)

out=conv(train[0][0],filter)

plt.imshow(out,cmap='gray')

np.save('gaussian.npy',out)
psnr(train[0][0],out)
#median filter

import numpy as np

def median(inp,ker):

    inp=pad1(inp)

    v_k,h_k=ker.shape

    v_i,h_i=inp.shape

    out=np.zeros((1+v_i-v_k,1+h_i-h_k))

    for i in range(v_i):

        for j in range(h_i):

            if (i+v_k<=v_i) and (j+h_k<=h_i):

                out[i,j]=np.median(inp[i:i+v_k,j:j+h_k].ravel())

    return out

out=median(train[0][0],np.zeros((3,3)))

plt.imshow(out,cmap='gray')

np.save('median.npy',out)
psnr(train[0][0],out)
#conservative filter

import numpy as np

def conservative(inp,ker):

    v_k,h_k=ker.shape

    v_i,h_i=inp.shape

    idx_v,idx_h=v_k//2,h_k//2

    out=np.zeros((1+v_i-v_k,1+h_i-h_k))

    output=inp.copy()

    for i in range(v_i):

        for j in range(h_i):

            nbr=[]

            for x in range(i-idx_v,i+idx_v):

                for y in range(j-idx_h,j+idx_h):

                    if (x>=0) and (y>=0) and (x<v_i) and (y<h_i):

                        nbr.append(inp[x,y])

            nbr.remove(inp[i,j])

            

            

            if inp[i,j]>max(nbr):

                output[i,j]=max(nbr)

            elif inp[i,j]<min(nbr):

                output[i,j]=min(nbr)

    return output

out=conservative(train[0][0],np.zeros((4,4)))

plt.imshow(out,cmap='gray')

np.save('conservative.npy',out)


psnr(train[0][0],out)
#laplacian filter

filter=np.array([[0,-1,0], [-1,4,-1], [0,-1,0]])

# filter/=np.sum(filter)

out=conv(train[0][0],filter)

plt.imshow(out,cmap='gray')

np.save('laplacian.npy',out)


psnr(train[0][0],out)
plt.imshow(out+train[0][0],cmap='gray')

np.save('laplacian1.npy',out+train[0][0])


psnr(train[0][0],out+train[0][0])
plt.imshow(np.where(out>0,1,0)+train[0][0],cmap='gray')

np.save('laplacian2.npy',np.where(out>0,1,0)+train[0][0])


psnr(train[0][0],np.where(out>0,1,0)+train[0][0])
#unsharp filter

filter=np.ones((3,3))

filter/=np.sum(filter)

out=conv(train[0][0],filter)

dff=train[0][0]-out

plt.imshow(train[0][0]+2*dff,cmap='gray')

np.save('unsharp.npy',train[0][0]+2*dff)


psnr(train[0][0],train[0][0]+2*dff)


dff=out-train[0][0]

plt.imshow(train[0][0]+2*dff,cmap='gray')

np.save('unsharp2.npy',train[0][0]+2*dff)


psnr(train[0][0],train[0][0]+2*dff)