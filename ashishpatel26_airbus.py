import numpy as np
import pandas as pd 
from skimage.data import imread
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img
from tqdm import tqdm_notebook
print(os.listdir("../input"))
Train_Image_folder='../input/train/'
Test_Image_folder='../input/test/'
Train_Image_name=os.listdir(path=Train_Image_folder)
Test_Image_name=os.listdir(path=Test_Image_folder)
Train_Image_path=[]
Train_Mask_path=[]
Train_id=[]
for i in Train_Image_name:
    path1=Train_Image_folder+i
    id1=i.split(sep='.')[0]
    Train_Image_path.append(path1)
    Train_id.append(id1)

df_Train_path=pd.DataFrame({'ImageId':Train_id,'Train_Image_path':Train_Image_path})
print('Train Shape: ',df_Train_path.shape)
df_Train_path.head()
Test_Image_path=[]
Test_id=[]
for i in Test_Image_name:
    path=Test_Image_folder+i
    id2=i.split(sep='.')[0]
    Test_Image_path.append(path)
    Test_id.append(id2)
df_Test_path=pd.DataFrame({'ImageId':Test_id,'Test_Image_path':Test_Image_path})
print('Test Shape: ',df_Test_path.shape)
df_Test_path.head()
masks = pd.read_csv('../input/train_ship_segmentations.csv')
print('Mask Shape: ',masks.shape)
masks.head()
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction
masks = pd.read_csv('../input/train_ship_segmentations.csv')
masks.head()