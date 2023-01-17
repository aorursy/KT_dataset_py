from keras.layers import Dense, Input 

from keras.layers import Conv2D, Flatten

from keras.layers import Reshape, Conv2DTranspose 

from keras.models import Model 

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint 

from keras.datasets import cifar10 

from keras.utils import plot_model 

from keras import backend as K 

import numpy as np 

import matplotlib.pyplot as plt 

import os

import cv2

import glob

def rgb_image(l, ab):

    shape = (l.shape[0],l.shape[1],3)

    img = np.zeros(shape)

    img[:,:,0] = l[:,:,0]

    img[:,:,1:]= ab

    img = img.astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def display(img):

    plt.figure()

    plt.set_cmap('gray')

    plt.imshow(img)

    plt.show()



imgs  = np.zeros((39000,128,128,3),dtype='uint8')



dirname=os.listdir('/kaggle/input/indoor-scenes-cvpr-2019/indoorCVPR_09/Images') 

i=0

for name in dirname:

    print(name)

    img_path = glob.glob('/kaggle/input/indoor-scenes-cvpr-2019/indoorCVPR_09/Images/'+name+'/*.jpg')

    for path in img_path:

        img=cv2.imread(path)

        if img is None:

            continue

        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img=cv2.resize(img,(128,128))

        imgs[i]=img

        i+=1



gray_scale = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')[:10000]

ab_scale = np.load('/kaggle/input/image-colorization/ab/ab/ab1.npy')[:10000]

save_dir = os.path.join(os.getcwd(), 'train') 





for i in range(10000):

    if i%1000==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(128,128))

    #lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    #cv2.imwrite(save_dir+'/'+str(i+1)+'.jpg', img)

    imgs[i+15000]=np.array(img)





gray_scale = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')[10000:20000]

ab_scale = np.load('/kaggle/input/image-colorization/ab/ab/ab2.npy')[:10000]

save_dir = os.path.join(os.getcwd(), 'train') 



if not os.path.isdir(save_dir):

    os.makedirs(save_dir)



for i in range(10000):

    if i%1000==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(128,128))

    imgs[i+25000]=np.array(img)







gray_scale = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')[20000:25000]

ab_scale = np.load('/kaggle/input/image-colorization/ab/ab/ab3.npy')[:5000]

save_dir = os.path.join(os.getcwd(), 'test') 



for i in range(4000):

    if i %1000==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(128,128))

    imgs[i+35000]=img

np.save('train128.npy',imgs)



imgs_test  = np.zeros((1000,128,128,3),dtype='uint8')

for i in range(4000,5000):

    if i%100==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(128,128))

    imgs_test[i-4000]=np.array(img)

np.save('test128.npy',imgs_test)
'''from keras.layers import Dense, Input 

from keras.layers import Conv2D, Flatten

from keras.layers import Reshape, Conv2DTranspose 

from keras.models import Model 

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint 

from keras.datasets import cifar10 

from keras.utils import plot_model 

from keras import backend as K 

import numpy as np 

import matplotlib.pyplot as plt 

import os

import cv2

def rgb_image(l, ab):

    shape = (l.shape[0],l.shape[1],3)

    img = np.zeros(shape)

    img[:,:,0] = l[:,:,0]

    img[:,:,1:]= ab

    img = img.astype('uint8')

    img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img

def display(img):

    plt.figure()

    plt.set_cmap('gray')

    plt.imshow(img)

    plt.show()



gray_scale = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')[:10000]

ab_scale = np.load('/kaggle/input/image-colorization/ab/ab/ab1.npy')[:10000]

save_dir = os.path.join(os.getcwd(), 'train') 





l_train  = np.zeros((20000,64,64,1),dtype='uint8')

ab_train  = np.zeros((20000,64,64,2),dtype='uint8')

for i in range(10000):

    if i%1000==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(64,64))

    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    ab_train[i]=lab[:,:,1:]

    l_train[i]=lab[:,:,0].reshape((64,64,1))

    #cv2.imwrite(save_dir+'/'+str(i+1)+'.jpg', img)



gray_scale = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')[10000:20000]

ab_scale = np.load('/kaggle/input/image-colorization/ab/ab/ab2.npy')[:10000]

for i in range(10000):

    if i%1000==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(64,64))

    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    ab_train[i+10000]=lab[:,:,1:]

    l_train[i+10000]=lab[:,:,0].reshape((64,64,1))



np.save('l_train64.npy',l_train)

np.save('ab_train64.npy',ab_train)



l_test  = np.zeros((5000,64,64,1),dtype='uint8')

ab_test = np.zeros((5000,64,64,2),dtype='uint8')

gray_scale = np.load('/kaggle/input/image-colorization/l/gray_scale.npy')[20000:25000]

ab_scale = np.load('/kaggle/input/image-colorization/ab/ab/ab3.npy')[:5000]

for i in range(5000):

    if i%1000==0:

        print(i)

    l_sample = gray_scale[i].reshape((224,224,1))

    ab_sample = ab_scale[i]

    img=rgb_image(l_sample,ab_sample)

    img=cv2.resize(img,(64,64))

    lab=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)

    ab_test[i]=lab[:,:,1:]

    l_test[i]=lab[:,:,0].reshape((64,64,1))



np.save('l_test64.npy',l_train)

np.save('ab_test64.npy',ab_train)'''