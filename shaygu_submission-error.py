import numpy as np, pandas as pd, os, gc

import matplotlib.pyplot as plt, time

from PIL import Image 

import warnings

warnings.filterwarnings("ignore")

import keras

from keras import backend as K

import os

import pandas as pd

import numpy as np

from skimage import morphology

import cv2

import glob



input_path = "../input"

list_dir_input = os.listdir(input_path)

print(list_dir_input)

list_dir_unet =  os.listdir("../input/" + list_dir_input[0])

print(list_dir_unet)

model_file =  list_dir_unet[0]

print(model_file)

model_path = input_path + '/' + list_dir_input[0] + '/' + model_file

print (model_path)

# model_path = "../input/unet-100-epochs/" + os.listdir("../input/unet-100-epochs")[0]









# print(os.listdir("../input/unettest3"))

# print(os.listdir("../input/unet-100-epochs"))

# print(os.listdir("../input/severstal-steel-defect-detection"))



# # os.listdir("../input/unet-100-epochs")[0]

# model_path = "../input/unet-100-epochs/" + os.listdir("../input/unet-100-epochs")[0]

# model_path2 = "../input/unettest3/" + os.listdir("../input/unettest3")[0]
img_h = 128

img_w = 800

channels = 3

path = '../input/severstal-steel-defect-detection/'

testfiles=os.listdir("../input/severstal-steel-defect-detection/test_images/")

print(len(testfiles))



# get all files using glob

test_files = [f for f in glob.glob('../input/severstal-steel-defect-detection/test_images/' + "*.jpg", recursive=True)]

print(len(test_files))
# # https://www.kaggle.com/xhlulu/severstal-simple-keras-u-net-boilerplate



# COMPETITION METRIC

def dice_coef(y_true, y_pred, smooth=1):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



# # LOAD MODEL



from keras.models import load_model

model = load_model(model_path,custom_objects={'dice_coef':dice_coef})

# model.summary()
from keras import backend as K

import numpy as np

import numpy as np, pandas as pd, os, gc

import matplotlib.pyplot as plt, time

from PIL import Image 

import warnings

import keras



warnings.filterwarnings("ignore")

path = '/kaggle/input/severstal-steel-defect-detection/'

train = pd.read_csv(path + 'train.csv')



# RESTRUCTURE TRAIN DATAFRAME

train['ImageId'] = train['ImageId_ClassId'].map(lambda x: x.split('.')[0]+'.jpg')

train2 = pd.DataFrame({'ImageId':train['ImageId'][::4]})

train2['e1'] = train['EncodedPixels'][::4].values

train2['e2'] = train['EncodedPixels'][1::4].values

train2['e3'] = train['EncodedPixels'][2::4].values

train2['e4'] = train['EncodedPixels'][3::4].values

train2.reset_index(inplace=True,drop=True)

train2.fillna('',inplace=True); 

train2['count'] = np.sum(train2.iloc[:,1:]!='',axis=1).values

train2.head(3)





class DataGenerator(keras.utils.Sequence):

    def __init__(self, df, batch_size = 16, subset="train", shuffle=False, 

                 preprocess=None, info={}):

        super().__init__()

        self.df = df

        self.shuffle = shuffle

        self.subset = subset

        self.batch_size = batch_size

        self.preprocess = preprocess

        self.info = info

        

        if self.subset == "train":

            self.data_path = path + 'train_images/'

        elif self.subset == "test":

            self.data_path = path + 'test_images/'

        self.on_epoch_end()



    def __len__(self):

        return int(np.floor(len(self.df) / self.batch_size))

    

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.df))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)

    

    def __getitem__(self, index): 

        X = np.empty((self.batch_size,128,800,3),dtype=np.float32)

        y = np.empty((self.batch_size,128,800,4),dtype=np.int8)

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):

            self.info[index*self.batch_size+i]=f

            X[i,] = Image.open(self.data_path + f).resize((800,128))

            if self.subset == 'train': 

                for j in range(4):

                    y[i,:,:,j] = rle2maskResize(self.df['e'+str(j+1)].iloc[indexes[i]])

        if self.preprocess!=None: X = self.preprocess(X)

        if self.subset == 'train': return X, y

        else: return X

        

# def rle2maskResize(rle):

#     # CONVERT RLE TO MASK 

#     if (pd.isnull(rle))|(rle==''): 

#         return np.zeros((128,800) ,dtype=np.uint8)

    

#     height= 256

#     width = 1600

#     mask= np.zeros( width*height ,dtype=np.uint8)



#     array = np.asarray([int(x) for x in rle.split()])

#     starts = array[0::2]-1

#     lengths = array[1::2]    

#     for index, start in enumerate(starts):

#         mask[int(start):int(start+lengths[index])] = 1

    

#     return mask.reshape( (height,width), order='F' )[::2,::2]





def preprocess(x):

    x = (x-110)/1.0

    return x
# https://www.kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode



def mask_to_rle(mask):

    '''

    Convert a mask into RLE

    

    Parameters: 

    mask (numpy.array): binary mask of numpy array where 1 - mask, 0 - background



    Returns: 

    sring: run length encoding 

    '''

    pixels= mask.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
from skimage import morphology



def remove_small_regions(img, size):

    """Morphologically removes small (less than size) connected regions of 0s or 1s."""

    img = morphology.remove_small_objects(img, size)

    img = morphology.remove_small_holes(img, size)

    return img



def process_pred_mask(pred_mask):

#     pred_mask = cv2.resize(pred_mask.astype('float32'),(1600, 256))

    pred_mask = (pred_mask > .8).astype(int)

    pred_mask = remove_small_regions(pred_mask, 15) * 255

    pred_mask = mask_to_rle(pred_mask)

    return pred_mask





from skimage.transform import rescale, resize, downscale_local_mean



test = pd.read_csv(path + 'sample_submission.csv')

test['ImageId'] = test['ImageId_ClassId'].map(lambda x: x.split('_')[0])

test_batches = DataGenerator(test.iloc[4:8],subset='test',batch_size=1,preprocess=preprocess)

test_preds = model.predict_generator(test_batches,verbose=1,steps = 1)



test_preds.shape




predictions = []

# submission = []

for image_ind in np.arange(0,1801):

    test_batches = DataGenerator(test.iloc[image_ind*4:(image_ind+1)*4],subset='test',batch_size=1,preprocess=preprocess)

    test_preds = model.predict_generator(test_batches,verbose=0,steps = 1)

    for mask_ind in np.arange(0,4):

        mask = test_preds[0,:,:,mask_ind]

        mask_resize = resize(mask,(256,1600))

        mask_post_process = process_pred_mask(mask_resize)

#         mask_rle = mask_to_rle(mask_resize)

        image_name = test['ImageId'].iloc[image_ind*4]

        name = image_name + f"_{mask_ind+1}"

#         [submission.append((image_name+'_%s' % (mask_ind + 1), mask_post_process))]

        predictions.append([name, mask_post_process])

    

df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])

df.to_csv("submission.csv", index=False)
print(df.head(20))
print(df.tail(15))
# # write it out

# # convert to a csv

# submission_df = pd.DataFrame(submission, columns=['ImageId_ClassId', 'EncodedPixels'])

# # check out some predictions and see if RLE looks ok

# submission_df[ submission_df['EncodedPixels'] != ''].head()





# submission_df.head()

# take a look at our submission 



# submission_df.tail(20)
# submission_df.to_csv('./submission.csv', index=False)
def process_pred_mask2(pred_mask):

#     pred_mask = cv2.resize(pred_mask.astype('float32'),(1600, 256))

    pred_mask = (pred_mask > .3).astype(int)

    pred_mask = remove_small_regions(pred_mask, 15) * 255

#     pred_mask = mask_to_rle(pred_mask)

    return pred_mask



batch_size = 16

img_resize_shape = (128,800)



# import glob 

def read_image(path_to_img):

    return Image.open("../input/severstal-steel-defect-detection/test_images/" + path_to_img).resize((img_resize_shape[1], img_resize_shape[0]))





import glob

# get all files using glob

test_files = [f for f in glob.glob('../input/severstal-steel-defect-detection/test_images/' + "*.jpg", recursive=True)]

plot_ind = 1

plt.figure(1,figsize = (80,80))

image_num = 10

image_init = 10

for image_ind in np.arange(image_init,image_init+image_num):

    img = np.array(read_image(test.iloc[::4]['ImageId'].values[image_ind]))

    img  = resize(img,(256,1600))

    plt.subplot(image_num,5,plot_ind)

    plt.imshow((img))

    plt.title(np.max(np.max(img)),fontsize = 50)

    plot_ind += 1

    for mask_ind in np.arange(0,4):

        test_batches = DataGenerator(test.iloc[image_ind*4:(image_ind+1)*4],subset='test',batch_size=1,preprocess=preprocess)

        test_preds = model.predict_generator(test_batches,verbose=1,steps = 1)

        mask = test_preds[0,:,:,mask_ind]

        mask_resize = resize(mask,(256,1600))

        plt.subplot(image_num,5,plot_ind)

        plt.imshow(mask_resize)

        plt.title(np.max(np.max(mask_resize)),fontsize = 50)

        plot_ind +=1
plot_ind = 1

plt.figure(1,figsize = (80,80))

image_num = 10

image_init = 0

for image_ind in np.arange(image_init,image_init+image_num):

    img = np.array(read_image(test.iloc[::4]['ImageId'].values[image_ind]))

    img  = resize(img,(256,1600))

    plt.subplot(image_num,5,plot_ind)

    plt.imshow((img))

    plt.title(np.max(np.max(img)),fontsize = 50)

    plot_ind += 1

    for mask_ind in np.arange(0,4):

        test_batches = DataGenerator(test.iloc[image_ind*4:(image_ind+1)*4],subset='test',batch_size=1,preprocess=preprocess)

        test_preds = model.predict_generator(test_batches,verbose=1,steps = 1)

        mask = test_preds[0,:,:,mask_ind]

        mask_resize = resize(mask,(256,1600))

        mask_postprocess = process_pred_mask2(mask_resize)

        plt.subplot(image_num,5,plot_ind)

        plt.imshow(mask_postprocess)

        plt.title(np.max(np.max(mask_postprocess)),fontsize = 50)

        plot_ind +=1
# When predicting all image to pred

# test_batches = DataGenerator(test.iloc[::4],subset='test',batch_size=1,preprocess=preprocess)

# test_preds = model.predict_generator(test_batches,verbose=1)





# import glob 

# def read_image(path_to_img):

#     return Image.open("../input/severstal-steel-defect-detection/test_images/" + path_to_img).resize((img_resize_shape[1], img_resize_shape[0]))





# plot_ind = 1

# plt.figure(1,figsize = (80,40))

# image_num = 5

# image_init = 100

# for image_ind in np.arange(image_init,image_init+image_num):

#     img = np.array(read_image(test.iloc[::4]['ImageId'].values[image_ind]))

#     plt.subplot(image_num,5,plot_ind)

#     plt.imshow((img)/255)

#     plot_ind += 1

#     for mask_ind in np.arange(0,4):

#         plt.subplot(image_num,5,plot_ind)

#         plt.imshow(test_preds[image_ind,:,:,mask_ind])

#         plt.title(np.max(np.max(test_preds[image_ind,:,:,mask_ind])),fontsize = 20)

#         plot_ind +=1






# plot_ind = 1

# plt.figure(1,figsize = (40,40))

# image_num = 10

# image_init = 500

# for image_ind in np.arange(image_init,image_init+image_num):

#     img = np.array(read_image(test.iloc[::4]['ImageId'].values[image_ind]))

#     img  = resize(img,(256,1600))

#     plt.subplot(image_num,5,plot_ind)

#     plt.imshow((img))

#     plot_ind += 1

#     for mask_ind in np.arange(0,4):

#         mask = test_preds[image_ind,:,:,mask_ind]

#         mask_resize = resize(mask,(256,1600))

#         mask_postprocess = process_pred_mask(mask_resize)

#         mask_rle = mask_to_rle(mask_postprocess)

#         rle_mask = rle2mask(mask_rle)

#         plt.subplot(image_num,5,plot_ind)

#         plt.imshow(rle_mask)

#         plt.title(np.max(np.max(rle_mask)),fontsize = 20)

#         plot_ind +=1

        

        