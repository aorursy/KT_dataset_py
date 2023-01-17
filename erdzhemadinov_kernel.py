# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import os

import albumentations as A

import torch

import torch.nn as nn

import pandas as pd

import numpy as np

import cv2

import gc

import time

from albumentations.pytorch import ToTensor

from torch.utils.data import DataLoader

from torch.autograd import Variable

from torch.optim import Adam,Adagrad,SGD

import torch.nn.functional as F

#optim.Adagrad 

from torchvision.models import resnet34,resnet50



import tqdm



#import warnings

import random



#warnings.filterwarnings("ignore")



#seed = 42

#random.seed(seed)

#os.environ["PYTHONHASHSEED"] = str(seed)



#np.random.seed(seed)



#torch.manual_seed(42)





from matplotlib import pyplot as plt

#import torch

#print(torch.__version__)

#print(torch.version.cuda)

#print(torch.backends.cudnn.version())

##print(torch.cuda.is_available())

#print(torch.cuda.device_count())

##print(torch.cuda.max_memory_allocated(device=None))

#print(torch.cuda.empty_cache())

#print(torch.backends.cuda.cufft_plan_cache.max_size)



#print(torch.backends.cuda.cufft_plan_cache.size)



#torch.backends.cuda.cufft_plan_cache.clear()
datamy = pd.read_csv('/kaggle/input/seismic/submission0.41.csv')



dataorg = pd.read_csv('/kaggle/input/seismic-pt2/submission0.55.csv')



datamybest = pd.read_csv('/kaggle/input/seismic/submission0.77.csv')





datapredict = pd.read_csv('/kaggle/input/lastdatasetrosneft/answers.csv')
datapredict.head()
datapredict = datapredict.drop(['EncodedPixels_x'], axis=1)



datapredict.columns = ['ImageId', 'ClassId', 'EncodedPixels']



datapredict.to_csv('answer.csv', index = None)
images = list(datamy.ImageId.unique())
def rle2mask(mask_rle, shape):

    """

    mask_rle: run-length as string formated (start length)ы

    shape: (width,height) of array to return

    Returns numpy array, 1 - mask, 0 - background

    """

    if mask_rle != mask_rle:

        return np.zeros_like(shape)



    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T





def mask2rle_another(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if b > prev + 1:

            run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return ' '.join(str(x) for x in run_lengths)
from  scipy import ndimage

direct = '/kaggle/input/seismic/seismic_test/img_for_users/'





def image(gray):

    

    

    

    return ndimage.binary_fill_holes(gray).astype(int)



def imageget(gray):

    

    

    

    return ndimage.binary_fill_holes(gray).astype(int)#).get()

def denoise(gray):

    

    

    img1 = gray

    return cv2.fastNlMeansDenoising( img1, 15.0, 2, 5)





def get_masks(id_image,cl):

    

    img = cv2.imread("{0}{1}".format(direct, id_image), 0)

    for i in range(0,7):

        if i ==cl:

            print(i)

            stringpr = list(datapredict[(datapredict.ImageId == id_image) & (datapredict.ClassId == i)]['EncodedPixels'].values)[0]



            stringmy = list(datamy[(datamy.ImageId == id_image) & (datamy.ClassId == i)]['EncodedPixels'].values)[0]

            stringorg = list(dataorg[(dataorg.ImageId == id_image) & (dataorg.ClassId == i)]['EncodedPixels'].values)[0]

            stringmybest = list(datamybest[(datamybest.ImageId == id_image) & (datamybest.ClassId == i)]['EncodedPixels'].values)[0]



            h, w = img.shape[0], img.shape[1]



            fig = plt.figure(figsize=(10,5))





            myimg = rle2mask(stringmy, (w, h))

            myimg1 = rle2mask(stringorg, (w, h) )

            myimg2 =  rle2mask(stringmybest, (w,h))

            myimg3 = imageget(rle2mask(stringmybest, (w,h)))

            myimg4  = denoise(rle2mask(stringmybest, (w,h)))

            myimg5 = rle2mask(stringpr, (w,h))







            #print(type(myimg3), type(myimg4))

            #print(myimg3.shape, myimg4.shape)

            #print(myimg3)

            #print(myimg4)

            kernel = np.ones((4,40), np.uint8)  # note this is a horizontal kernel

            d_im = cv2.dilate(np.uint8(myimg5), kernel, iterations=2)

            #if np.mean( myimg  ) < 0.024: 

            if np.mean( myimg  ) < 0.13: 

                d_im = d_im

            else:

                d_im = np.uint8(myimg5) #cv2.dilate(np.uint8(myimg3), kernel, iterations=1)

            #e_im = cv2.erode(d_im, kernel, iterations=1) 

            print(np.mean( myimg ))         

            

            



            #myimg4  = denoise(rle2mask(stringmybest, (w,h))



            #print(type(e_im))

            #print(type(myimg3))



            images = [myimg, myimg1,myimg2,img, myimg3,myimg4, d_im,  myimg5]

            imname = ['My ~ 0.41 DICE', 'Boosters ~ 0.55 DICE','Mybest ~ 0.77 DICE',

                      'Original', 'Fixed','Denoise','lines', 'Show', ]

            fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24,8))

            for i, ax in enumerate(axs.flatten()):

                #print(imname[i])

                plt.sca(ax)

                plt.imshow(images[i])



                plt.title('Image: {}'.format(imname[i]))



            plt.suptitle('Overall Title')

            plt.show()



    

    print(img.shape)



        

    return None





[get_masks(	i ,3	) for i in random.choices(images, k=5) ]
!pip install swifter



import swifter
cnt =0 

def fix_bugs(x):

    global cnt

    cnt+=1

    if cnt%500==0:

        print(cnt/len(datamybest ))

        

    

    if x['ClassId'] ==6:

        img = cv2.imread("{0}{1}".format(direct, x['ImageId']), 0)

    #for i in range(0,7):

     #   if i ==cl:

    #        print(i)

    #        stringpr = list(datapredict[(datapredict.ImageId == id_image) & (datapredict.ClassId == i)]['EncodedPixels'].values)[0]

    #print("{0}{1}".format(direct, x['ImageId']))

    #img = cv2.imread("{0}{1}".format(direct, x['ImageId']), 0)

    

    #print("{0}{1}".format(direct, x['ImageId']))

#         h, w = img.shape[0], img.shape[1]

#         kernel = np.ones((2,40), np.uint8) 

#         d_im = cv2.dilate(np.uint8(image(rle2mask(x['EncodedPixels'], (w, h) ))), kernel, iterations=2)

        #fig = plt.figure(figsize=(10,5))



        h, w = img.shape[0], img.shape[1]



        #fig = plt.figure(figsize=(10,5))





        myimg = rle2mask(x['EncodedPixels'], (w, h))

        

        

        kernel = np.ones((2,40), np.uint8)  # note this is a horizontal kernel

        d_im = cv2.dilate(np.uint8(myimg), kernel, iterations=1)

        #if np.mean( myimg  ) < 0.024: 

        if np.mean( myimg  ) < 0.024: 

            d_im = d_im

            x['EncodedPixels'] = mask2rle_another(d_im)

        else:

            d_im = np.uint8(myimg)

            

            

            

            

#         images = [myimg, d_im]

#         imname = ['My ~ 0.77 DICE', 'Boosters ~ 0.77 FIXED' ]

#         fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(24,8))

#         for i, ax in enumerate(axs.flatten()):

#             #print(imname[i])

#             plt.sca(ax)

#             plt.imshow(images[i])



#             plt.title('Image: {}'.format(imname[i]))



#         plt.suptitle('Overall Title')

#         plt.show()





#     if np.mean(d_im ) < 0.1: 

#         #d_im = d_im

#         x['EncodedPixels'] = mask2rle_another(denoise(np.uint8(image(rle2mask(x['EncodedPixels'], (w, h) )))))

#     else:

#         #d_im = np.uint8(myimg3)

#         x['EncodedPixels'] = mask2rle_another(image(rle2mask(x['EncodedPixels'], (w, h) )))

    

    return x



datapredict = datapredict.swifter.apply(lambda x: fix_bugs(x), axis=1)



datapredict.to_csv("thirdlay.csv", index= None)


direct
cnt = 0





def fix_bugs(x):

    global cnt

    cnt+=1

    if cnt%200:

        print(cnt/len(datamybest ))

    #print("{0}{1}".format(direct, x['ImageId']))

    img = cv2.imread("{0}{1}".format(direct, x['ImageId']), 0)

    

    #print("{0}{1}".format(direct, x['ImageId']))

    h, w = img.shape[0], img.shape[1]

    kernel = np.ones((2,40), np.uint8) 

    d_im = cv2.dilate(np.uint8(image(rle2mask(x['EncodedPixels'], (w, h) ))), kernel, iterations=1)

    if np.mean(d_im ) < 0.1: 

        #d_im = d_im

        x['EncodedPixels'] = mask2rle_another(denoise(np.uint8(image(rle2mask(x['EncodedPixels'], (w, h) )))))

    else:

        #d_im = np.uint8(myimg3)

        x['EncodedPixels'] = mask2rle_another(image(rle2mask(x['EncodedPixels'], (w, h) )))

    

    return x



#datamybest = datamybest.swifter.apply(lambda x: fix_bugs(x), axis=1)

import os

import cv2

import numpy as np

from statistics import mode

import pandas as pd

import  matplotlib

from matplotlib import cm

#print(dir(matplotlib))

import random

from matplotlib import pyplot as plt
# masks = pd.read_csv('/kaggle/input/seismic/seismic_csv.csv')

# rles = {}

# print(masks.head())



# wp = '/ready_data/'

# wpmask  = '/ready_data_mask/'





# image_names = []

# image_width =[]

# image_rle = []



# try:

#     os.mkdir(wp)

# except OSError:

#     print("Creation of the directory %s failed" % wp)

# else:

#     print("Successfully created the directory %s " % wp)



# try:

#     os.mkdir(wpmask)

# except OSError:

#     print("Creation of the directory %s failed" % wp)

# else:

#     print("Successfully created the directory %s " % wp)    



# _path_output = '/kaggle/input/seismic/seismic_stage2/train/masks/'



# _path_mask = '/kaggle/input/seismic/seismic_stage2/train/images/'





# _list_out = os.listdir(path=_path_output)



# # print('Train count' , len(_list_in))

# print('Test count',  len(_list_out))

# _list_out 
# def size_analyse(x):

#     spisok =[]

#     count = 0







#     img = None

#     for i in x:

#         print(i)

#         if i.split("_")[0] == 'inline':

#             w = 384



#         else:

#             w = 512







#         if count % 100 == 0:

#             print(count / len(x))

#         count+=1





#             #print("{0}{1}".format(_path_output, i))

#         img = cv2.imread("{0}{1}".format(_path_output, i), 1)

#         height, width = img.shape[:2]

#         pt = 0





#         masks_im = []

#         for j in range(0, 7):

#             #imgmask = cv2.imread("{0}{1}".format(_path_mask, i), 1)

#             #'ImageId': ImageId, 'ClassId': ClassId, 'EncodedPixels'





#             masks_im.append(rle2mask(

#                 list(masks[(masks.ImageId == i) & (masks.ClassId == j) ]

#                      ['EncodedPixels'].values)[0], (width, height)))



#             #print(type(rle2mask(

#              #   list(masks[(masks.ImageId == i) & (masks.ClassId == j) ]

#               #       ['EncodedPixels'].values)[0], (width, height))))

#             #print(rle2mask(

#              #   list(masks[(masks.ImageId == i) & (masks.ClassId == j) ]

#               #       ['EncodedPixels'].values)[0], (width, height)).shape)

#         #print(masks_im)







#         # else:

#         # #   img = cv2.imread("{0}{1}".format(_path_input, i), 0)



#         if width >= w:

#             #wpmask

#             pt  = 0

#             cnt = 0

#             #print("width {0} ||| cnt {1} |||| width- cnt {2}|||| imagename {3}".format(width, cnt, width - cnt, i))

#             while (width - cnt) >= w:

#                 print("width {0} ||| cnt {1} |||| width- cnt {2}|||| imagename {3}".format(width, cnt, width-cnt, i))

#                 ext = i.split(".")

#                 if (width - cnt) >= 2*w:

#                     #print(width - cnt)

#                     #print("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), cnt,cnt+w)

#                     #matplotlib.image.imsave("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:cnt + w],

#                      #                       cmap='gray')



#                     plt.imsave("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:cnt + w],

#                                cmap='gray')



#                     image_names.append("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]))



#                     #plt.imsave("{0}{1}_{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:cnt + w],

#                      #                      cmap='gray')



#                     #image_names.append("{0}{1}_{2}.{3}".format(wp, ext[0], pt, ext[1]))

#                     image_width.append(width)

                    







#                     #plt.imsave("{0}{1}_{2}.{3}".format(wpmask, ext[0], pt, ext[1]), imgmask[:, cnt:cnt + w],

#                     #                       cmap='gray')



#                     #image_names.append("{0}{1}_{2}.{3}".format(wpmask, ext[0], pt, ext[1]))

#                     #image_width.append(width)



#                     for j in range(0, 7):

#                         # imgmask = cv2.imread("{0}{1}".format(_path_mask, i), 1)

#                         # 'ImageId': ImageId, 'ClassId': ClassId, 'EncodedPixels'

#                         mphoto = masks_im[j]

#                         rles['{0}{1}'.format("{0}{1}{2}.{3}".format(wp,

#                                                                     ext[0],

#                                                                     pt,

#                                                                     ext[1]),

#                                              j)] = mask2rle_another( mphoto[:, cnt:cnt + w])

#                             #.append(mask2rle_another(

#                             #list(masks[(masks.ImageId == j) & (masks.ClassId == i)]

#                             #     ['EncodedPixels'].value)[0]))

#                     #image_rle.append(mask2rle_another(imgmask[:, cnt:cnt + w]))



#                     #matplotlib.ima

#                     #cv2.imwrite("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:cnt+w])



#                 else:

#                     #cv2.imwrite("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:])





#                     #plt.imsave("{0}{1}_{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:])

#                     #image_names.append("{0}{1}_{2}.{3}".format(wp, ext[0], pt, ext[1]))



#                     plt.imsave("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), img[:, cnt:])

#                     image_names.append("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]))

#                     image_width.append(width)



#                     for j in range(0, 7):

#                         # imgmask = cv2.imread("{0}{1}".format(_path_mask, i), 1)

#                         # 'ImageId': ImageId, 'ClassId': ClassId, 'EncodedPixels'

#                         #print(masks_im[j])

#                         mphoto = masks_im[j]

#                         #print(mphoto[:, cnt:].shape)

#                         #print(masks_im[i].shape)

#                         rles['{0}{1}'.format("{0}{1}{2}.{3}".format(wp,

#                                                      ext[0],

#                                                      pt,

#                                                      ext[1]), j)] = mask2rle_another( mphoto[:, cnt:])

#                         # .



                    

                    

                    

#                     break

#                 cnt += w

#                 pt += 1

#         else:

#             ext = i.split(".")

#             plt.imsave("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]), img)



#             image_names.append("{0}{1}{2}.{3}".format(wp, ext[0], pt, ext[1]))

#             image_width.append(width)

            



            

#             #plt.imsave("{0}{1}_{2}.{3}".format(wpmask, ext[0], pt, ext[1]), imgmask,

#              #                              cmap='gray')



#             #image_rle.append(mask2rle_another(imgmask))

#             for j in range(0, 7):

#                 # imgmask = cv2.imread("{0}{1}".format(_path_mask, i), 1)

#                 # 'ImageId': ImageId, 'ClassId': ClassId, 'EncodedPixels'



#                 rles['{0}{1}'.format("{0}{1}{2}.{3}".format(wp,

#                                                             ext[0],

#                                                             pt,

#                                                             ext[1]), j)]  = mask2rle_another( masks_im[i][:, :])







# size_analyse(_list_out)





# ImageId = []

# ClassId = []

# EncodedPixels =[]

# ImageWidth = []

# cnt=0

# for i in image_names:



#     for j in range(0, 7):

#         ImageId.append(i.split("\\")[-1])

#         ClassId.append(j)

#         ##EncodedPixels.append("{0} {1}".format(random.randint(100, 300), random.randint(5, 40)))

#         #EncodedPixels.append(image_rle[cnt])

#         EncodedPixels.append(rles["{0}{1}".format(i,j)])

#         ImageWidth.append(image_width[cnt])

#     cnt+=1

# print(len(ImageId),len(ClassId), len(EncodedPixels),len(ImageWidth))

# #d = {'ImageId': ImageId, 'ClassId': ClassId, 'EncodedPixels': EncodedPixels, 'Width': ImageWidth}

# d = {'ImageId': ImageId, 'ClassId': ClassId, 'EncodedPixels': EncodedPixels}

# output = pd.DataFrame(data=d)

# output.to_csv('adding_data.csv', index=None)







#import pandas as pd

#seismic_csv = pd.read_csv("../input/seismic/seismic_csv.csv")

#submission0_41 = pd.read_csv("../input/seismic/submission0.41.csv")

#submission0_77 = pd.read_csv("../input/seismic/submission0.77.csv")
#import pandas as pd#

#submission0_55 = pd.read_csv("../input/seismic-pt2/submission0.55.csv")