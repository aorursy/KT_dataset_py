import os

import sys

import random



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



from skimage.transform import resize

from skimage.morphology import label

from skimage.feature import hog

from skimage import exposure

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.feature import canny

from skimage.filters import sobel

from skimage.morphology import watershed

from scipy import ndimage as ndi

import warnings

warnings.filterwarnings("ignore")

from skimage.segmentation import mark_boundaries

from scipy import signal

import cv2

import glob, pylab, pandas as pd

import pydicom, numpy as np

import tqdm

import gc

gc.enable()

import glob



from skimage.transform import resize

from skimage.morphology import label

from skimage.feature import hog

from skimage import exposure

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from skimage.feature import canny

from skimage.filters import sobel

from skimage.morphology import watershed

from scipy import ndimage as ndi

import warnings

warnings.filterwarnings("ignore")

from skimage.segmentation import mark_boundaries



import sys

sys.path.insert(0, '../input/siim-acr-pneumothorax-segmentation/')

from mask_functions import rle2mask
os.listdir('../input')
sample_files = glob.glob('../input/siim-acr-pneumothorax-segmentation/sample images/*.dcm')

len(sample_files)
df = pd.read_csv('../input/siim-acr-pneumothorax-segmentation/sample images/train-rle-sample.csv',header=None)

df.head()
import cv2

from IPython.display import display, Image

def cvshow(image, format='.png', rate=255 ):

    decoded_bytes = cv2.imencode(format, image*rate)[1].tobytes()

    display(Image(data=decoded_bytes))

    return
j = 0

nImg = 10

img_ar = np.empty(0)

while img_ar.shape[0]!=nImg:

#     dcm_file = '../input/../input/sample images/%s.dcm' % patientId

    dcm_file = sample_files[j]

    dcm_data = pydicom.read_file(dcm_file)

    img = np.expand_dims(dcm_data.pixel_array,axis=0)    

    if j==0:

        img_ar = img

    elif (j%100==0):

        print(j,'images loaded')

    else:

        img_ar = np.concatenate([img_ar,img],axis=0)

    j += 1

    
def imgtile(imgs,tile_w):

    assert imgs.shape[0]%tile_w==0,"'imgs' cannot divide by 'th'."

    r=imgs.reshape((-1,tile_w)+imgs.shape[1:])

    return np.hstack(np.hstack(r))



#usage

tiled = imgtile(img_ar,5)

# cvshow(tiled)

tiled.shape
cvshow(cv2.resize( tiled, (1024,512), interpolation=cv2.INTER_LINEAR ))
# simple ways to enhance the image



plt.figure(figsize=(30,15))

plt.subplots_adjust(bottom=0.2, top=0.7, hspace=0)  #adjust this to change vertical and horiz. spacings..

nImg = 3  #no. of images to process

for j in range(nImg):

    q = j+1

    img = np.array(pydicom.read_file(sample_files[j]).pixel_array)

    

#     # Contrast stretching

    p2, p97 = np.percentile(img, (2, 97))

    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p97))

    

    # Equalization

    img_eq = exposure.equalize_hist(img)



    # Adaptive Equalization

    img_adapteq = exposure.equalize_adapthist(img)

    img_adjustlog = exposure.adjust_log(img)

    img_adjustgamma = exposure.adjust_gamma(img)

    

    plt.subplot(nImg,7,q*7-6)

    plt.imshow(img, cmap=plt.cm.bone)

    plt.title('Original Image')

    

    

    plt.subplot(nImg,7,q*7-5)    

    plt.imshow(img_rescale, cmap=plt.cm.bone)

    plt.title('Contrast stretching')

    

    

    plt.subplot(nImg,7,q*7-4)

    plt.imshow(img_eq, cmap=plt.cm.bone)

    plt.title('Equalization')

    

    

    plt.subplot(nImg,7,q*7-3)

    plt.imshow(img_adapteq, cmap=plt.cm.bone)

    plt.title('Adaptive Equalization')

    

    

    plt.subplot(nImg,7,q*7-2)

    plt.imshow(img_adjustgamma, cmap=plt.cm.bone)

    plt.title('Adjust Gamma')

    

    plt.subplot(nImg,7,q*7-2)

    plt.imshow(img_adjustlog, cmap=plt.cm.bone)

    plt.title('Adjust Log')







plt.show()
train_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-train/*/*/*.dcm'

test_glob = '../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/dicom-images-test/*/*/*.dcm'

train_fns = sorted(glob.glob(train_glob))

test_fns = sorted(glob.glob(test_glob))

df_mask = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv', index_col='ImageId')
len(df_mask),len(set(df_mask.index.values))
df_mask_dist = df_mask.copy(deep=True)

df_mask_dist.reset_index(inplace=True)



df = pd.DataFrame(df_mask_dist.ImageId.value_counts()).reset_index()

df.columns = ['ImageId','counts']



df_mask_dist = pd.merge(df_mask_dist,df,on='ImageId')

df_mask_dist.head()
df_mask_dist.counts.plot.hist()

plt.xlabel('Number of Annotations')
def get_image_data(im_path,im_type,df_mask=df_mask):

    if im_type=='train':

        return pydicom.dcmread(im_path).pixel_array
train_fn_id = [val.split('/')[-1][:-4] for val in train_fns]

print(len(set(train_fn_id).intersection(set(df_mask_dist.ImageId))))
masks_count_gt_one = set(df_mask_dist[df_mask_dist.counts>4].ImageId.values)

print('Number of masks with annotations greater than 5:',(df_mask_dist.counts>5).sum())

common_masks =  list(set(train_fn_id).intersection(masks_count_gt_one))

print(len(common_masks))
nImg = 12  #no. of images that you want to display

np.random.seed(42)

sel_train_fns = [train_fns[train_fn_id.index(c_m)] for c_m in common_masks]



_train_ids = list(sel_train_fns)

np.random.shuffle(_train_ids)

_train_ids = _train_ids[:nImg]

tile_size = (256, 256)

n = 4

alpha = 0.25



m = int(np.ceil(len(_train_ids) * 1.0 / n))

complete_image = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 1), dtype=np.uint8)

complete_image_masked = np.zeros((m*(tile_size[0]+2), n*(tile_size[1]+2), 1), dtype=np.uint8)



counter = 0

for i in range(m):

    ys = i*(tile_size[1] + 2)

    ye = ys + tile_size[1]

    for j in range(n):

        xs = j*(tile_size[0] + 2)

        xe = xs + tile_size[0]

        if counter == len(_train_ids):

            break

        image_path = _train_ids[counter]; counter+=1

        img = get_image_data(image_path, 'train')

        

        num_annot = len(df_mask.loc[image_path.split('/')[-1][:-4]].values)

#         print('num_annot',num_annot)

        for i in range(num_annot):

            mask =  rle2mask(df_mask.loc[image_path.split('/')[-1][:-4]].iloc[i][0], 1024, 1024).T

            

        mask = mask.astype(np.uint8)

        

        img_masked =  cv2.addWeighted(img, alpha, mask, 1 - alpha,0)

#         img_masked = cv2.bitwise_and(img, img, mask=mask)



        img = cv2.resize(img, dsize=tile_size)

        img_masked = cv2.resize(img_masked, dsize=tile_size)

        

        img_masked = cv2.putText(img_masked, image_path.split('/')[-1][:-4], (5,img.shape[0] - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), thickness=2)

        complete_image_masked[ys:ye, xs:xe, :] = img_masked[:,:,None]

        

    if counter == len(_train_ids):

        break    

    

m = complete_image.shape[0] / (tile_size[0] + 2)

k = 8

n = int(np.ceil(m / k))

for i in range(n):

    plt.figure(figsize=(20, 20))

    ys = i*(tile_size[0] + 2)*k

    ye = min((i+1)*(tile_size[0] + 2)*k, complete_image.shape[0])

    plt.imshow(complete_image_masked[ys:ye,:,0],cmap='bone')

    plt.title("Training dataset: Lighter Color depicts mask")