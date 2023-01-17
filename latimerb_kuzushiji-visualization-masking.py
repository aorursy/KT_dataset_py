# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

!mkdir train_images_mask

print(os.listdir("../input/kuzushiji-recognition"))
df = pd.read_csv("../input/kuzushiji-recognition/train.csv")

df.head()
import glob

from skimage.io import imread

from skimage.color import rgb2gray





example_file = glob.glob("../input/kuzushiji-recognition/train_images/{}.jpg".format(df.iloc[0,0]))[0]

im = imread(example_file,as_gray=True)



plt.figure(figsize=(10,10))

plt.imshow(im,cmap='gray')



#print("../input/train_images/{}.jpg".format(df.iloc[0,0]))
char_locs = df.iloc[0,1].split()



# Reshape into matrix

char_locs = np.reshape(np.asarray(char_locs), (len(char_locs)//5, 5))

char_unicode = char_locs[:,0]

char_locs = char_locs[:,1:].astype(np.int)
fig = plt.figure(figsize=(20,10))

plt.subplot(1,4,1)

plt.imshow(im,'gray')



for i in np.arange(char_locs.shape[0]):



    x = char_locs[i,0]

    y = char_locs[i,1]

    w = char_locs[i,2]

    h = char_locs[i,3]



    plt.plot([x,x+w],[y,y],'r')

    plt.plot([x,x+w],[y+h,y+h],'r')

    plt.plot([x,x],[y,y+h],'r')

    plt.plot([x+w,x+w],[y,y+h],'r')

    

r_samps = np.random.randint(0,char_locs.shape[0],9)

sps = [2,3,4,6,7,8,10,11,12]

for i in np.arange(0,r_samps.shape[0]):

    plt.subplot(3,4,sps[i])

    x = char_locs[r_samps[i],0]

    y = char_locs[r_samps[i],1]

    w = char_locs[r_samps[i],2]

    h = char_locs[r_samps[i],3]

    plt.imshow(im[y:y+h,x:x+w],cmap='gray')
# Now to detect the characters

# inspired by https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
x = char_locs[50,0]

y = char_locs[50,1]

w = char_locs[50,2]

h = char_locs[50,3]



z_im = (im[y:y+h,x:x+w]-np.mean(im[y:y+h,x:x+w]))/np.std(im[y:y+h,x:x+w])



plt.figure(figsize=(10,10))

plt.subplot(1,2,1)

plt.imshow(z_im,cmap='gray', vmin=np.amin(z_im), vmax=np.amax(z_im))

plt.subplot(1,2,2)

im_mask = np.copy(im[y:y+h,x:x+w])

im_mask[z_im>0] = 0

im_mask[z_im<=0] = 1

plt.imshow(im_mask,cmap='gray',vmin=0,vmax=1.)
im_mask_whole = np.zeros((im.shape[0],im.shape[1]))



for i in np.arange(0,char_locs.shape[0]):

    x = char_locs[i,0]

    y = char_locs[i,1]

    w = char_locs[i,2]

    h = char_locs[i,3]



    z_im = (im[y:y+h,x:x+w]-np.mean(im[y:y+h,x:x+w]))/np.std(im[y:y+h,x:x+w])



    im_mask = np.copy(im[y:y+h,x:x+w])

    im_mask[z_im>0] = 0

    im_mask[z_im<=0] = 1

    

    im_mask_whole[y:y+h,x:x+w] = im_mask
plt.figure(figsize=(10,10))

plt.imshow(im_mask_whole,cmap='gray',vmin=0,vmax=1.)
import imageio

imageio.imwrite('./train_images_mask/mask_{}.jpg'.format(df.iloc[0,0]), im_mask_whole)
def make_masks(filename_df):

    

    for i in np.arange(0,filename_df.shape[0]):

        # Load the file

        file = glob.glob("../input/kuzushiji-recognition/train_images/{}.jpg".format(df.iloc[i,0]))[0]

        im = imread(file,as_gray=True)

        # Make im_mask_whole to store the masked image

        im_mask_whole = np.zeros((im.shape[0],im.shape[1]))

        # If the image is not NaN, continue

        if not(pd.isnull(filename_df.iloc[i,1])):

            

            # Make a list of the character names (unicode) and locations

            char_locs = filename_df.iloc[i,1].split()

            # Reshape into matrix

            char_locs = np.reshape(np.asarray(char_locs), (len(char_locs)//5, 5))

            char_unicode = char_locs[:,0]

            # Locations are integers

            char_locs = char_locs[:,1:].astype(np.int)

        

            for j in np.arange(0,char_locs.shape[0]):

                x = char_locs[j,0]

                y = char_locs[j,1]

                w = char_locs[j,2]

                h = char_locs[j,3]



                z_im = (im[y:y+h,x:x+w]-np.mean(im[y:y+h,x:x+w]))/np.std(im[y:y+h,x:x+w])



                im_mask = np.copy(im[y:y+h,x:x+w])

                im_mask[z_im>0] = 0

                im_mask[z_im<=0] = 1

    

                im_mask_whole[y:y+h,x:x+w] = im_mask

        

            #imageio.imwrite('./train_images_mask/mask_{}.jpg'.format(df.iloc[i,0]), im_mask_whole)

        

#make_masks(df)

        
#file = glob.glob("../input/kuzushiji-recognition/train_images/{}.jpg".format(df.iloc[200,0]))[0]

#file_mask = glob.glob("./train_images_mask/mask_{}.jpg".format(df.iloc[200,0]))[0]

#im = imread(file,as_gray=True)

#im_mask = imread(file_mask,as_gray=True)



#plt.figure(figsize=(10,10))

#plt.subplot(1,2,1)

#plt.imshow(im,cmap='gray')

#plt.subplot(1,2,2)

#plt.imshow(im_mask,cmap='gray')
import os

import numpy as np

import torch

import torch.utils.data

from PIL import Image



class KuzushijiDataset(torch.utils.data.Dataset):

    def __init__(self, root, transforms=None):

        self.root = root

        self.transforms = transforms

        # load all image files, sorting them to

        # ensure that they are aligned

        self.imgs = list(sorted(os.listdir(os.path.join(root, "../input/kuzushiji-recognition/train_images"))))

        self.masks = list(sorted(os.listdir(os.path.join(root, "train_images_mask"))))



    def __getitem__(self, idx):

        # load images ad masks

        img_path = os.path.join(self.root, "../input/kuzushiji-recognition/train_images", self.imgs[idx])

        mask_path = os.path.join(self.root, "train_images_mask", self.masks[idx])

        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,

        # because each color corresponds to a different instance

        # with 0 being background

        mask = Image.open(mask_path)



        mask = np.array(mask)

        # instances are encoded as different colors

        obj_ids = np.unique(mask)

        # first id is the background, so remove it

        obj_ids = obj_ids[1:]



        # split the color-encoded mask into a set

        # of binary masks

        masks = mask == obj_ids[:, None, None]



        # get bounding box coordinates for each mask

        num_objs = len(obj_ids)

        boxes = []

        for i in range(num_objs):

            pos = np.where(masks[i])

            xmin = np.min(pos[1])

            xmax = np.max(pos[1])

            ymin = np.min(pos[0])

            ymax = np.max(pos[0])

            boxes.append([xmin, ymin, xmax, ymax])



        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class

        labels = torch.ones((num_objs,), dtype=torch.int64)

        masks = torch.as_tensor(masks, dtype=torch.uint8)



        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["masks"] = masks

        target["image_id"] = image_id

        target["area"] = area

        target["iscrowd"] = iscrowd



        if self.transforms is not None:

            img, target = self.transforms(img, target)



        return img, target



    def __len__(self):

        return len(self.imgs)
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor





def get_model_instance_segmentation(num_classes):

    # load an instance segmentation model pre-trained pre-trained on COCO

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)



    # get number of input features for the classifier

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    # now get the number of input features for the mask classifier

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    hidden_layer = 256

    # and replace the mask predictor with a new one

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,

                                                       hidden_layer,

                                                       num_classes)



    return model