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
#https://www.kaggle.com/inversion/run-length-decoding-quick-start
ImageId = '0005d01c8.jpg'

img = imread('../input/train/' + ImageId)
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
