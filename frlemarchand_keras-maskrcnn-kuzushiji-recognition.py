import os

import gc

import sys

import time

import json

import glob

import random

from pathlib import Path

import pandas as pd



from PIL import Image

import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from imgaug import augmenters as iaa



import itertools

from tqdm import tqdm
training_df = pd.read_csv("../input/rle-kuzushiji-dataset-57-most-freq-symbols/RLE_dataset_57most_freq_symbols.csv")

# training_df = pd.read_csv("../input/rle-kuzushiji-dataset-649-most-freq-symbols/dataset_with_rle_649classes.csv")

unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/kuzushiji-recognition/unicode_translation.csv').values}
frequency_dict = {}

boundaries_dict = {}
for idx, row in training_df.iterrows():

    try:

        page_labels = np.array(row["labels"].split(' ')).reshape(-1, 5)

    except:

        pass

    for symbol_info in page_labels:

        symbol_unicode = symbol_info[0]

        x, y, boundary_width, boundary_height  = [int(item) for item in symbol_info[1:]]

        if not frequency_dict.get(symbol_unicode):

            frequency_dict[symbol_unicode] = 1

            boundaries_dict[symbol_unicode] = [boundary_width*boundary_height]

        else:

            frequency_dict[symbol_unicode] += 1

            boundaries_dict[symbol_unicode].append(boundary_width*boundary_height)
plt.plot(frequency_dict.values())

plt.show()
count=0

total_samples=0

threshold=2000

for kuzushiji_count in frequency_dict.values():

    if kuzushiji_count>threshold:

        count+=1

        total_samples+=kuzushiji_count

print("Number of classes kept:",count)

print("Percentage of the examples these classes represent:",100*total_samples/sum(frequency_dict.values()))
kuzushiji_to_detect = []

for unicode, kuzushiji_count in frequency_dict.items():

    if kuzushiji_count>threshold:

        kuzushiji_to_detect.append(unicode)

        

#Added the sort as the model would not seem to learn and only learn background masks instead of the actual classes' masks.

kuzushiji_to_detect = sorted(kuzushiji_to_detect)
DATA_DIR = Path('../kaggle/input')

ROOT_DIR = "../../working"



NUM_CATS = len(kuzushiji_to_detect)

IMAGE_SIZE = 512
!git clone https://www.github.com/matterport/Mask_RCNN.git

os.chdir('Mask_RCNN')



!rm -rf .git # to prevent an error when the kernel is committed

!rm -rf images assets # to prevent displaying images at the bottom of a kernel
sys.path.append(ROOT_DIR+'/Mask_RCNN')

from mrcnn.config import Config



from mrcnn import utils

import mrcnn.model as modellib

from mrcnn import visualize

from mrcnn.model import log
!wget --quiet https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

!ls -lh mask_rcnn_coco.h5



COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
class KuzushijiConfig(Config):

    NAME = "kuzushiji"

    NUM_CLASSES = NUM_CATS + 1 # +1 for the background class

    

    GPU_COUNT = 1

    IMAGES_PER_GPU = 4 # a memory error occurs when IMAGES_PER_GPU is too high

    

    BACKBONE = 'resnet50'

    

    IMAGE_MIN_DIM = IMAGE_SIZE

    IMAGE_MAX_DIM = IMAGE_SIZE    

    IMAGE_RESIZE_MODE = 'none'

    

    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    

    # STEPS_PER_EPOCH should be the number of instances 

    # divided by (GPU_COUNT*IMAGES_PER_GPU), and so should VALIDATION_STEPS;

    # however, due to the time limit, I set them so that this kernel can be run in 9 hours

    STEPS_PER_EPOCH = 360

    VALIDATION_STEPS = 40

    

config = KuzushijiConfig()

config.display()
category_dict = {}

unique_symbols = kuzushiji_to_detect

for category_id, key in enumerate(unique_symbols):

    category_dict[key] = category_id
training_df.head()
#Apply larger dilation until the mask is in one block

def get_unique_mask(cropped_mask):

    is_dilation_complete = False

    cropped_mask = cropped_mask.astype("uint8")

    

    #Check if the current mask embeds all features in one "polygon"

    contours= cv2.findContours(cropped_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    if len(contours[0])==1:

        is_dilation_complete = True

        #just a bit of dilation to make the mask smoother

        kernel = np.ones((5,5),np.uint8)

        dilation = cv2.dilate(cropped_mask,kernel,iterations = 1)

    

    #Otherwise, let's dilate the mask until it embeds all features

    kernel_factor = 1

    while not is_dilation_complete:

        kernel_size = kernel_factor*5

        kernel = np.ones((kernel_size,kernel_size),np.uint8)

        dilation = cv2.dilate(cropped_mask,kernel,iterations = 1)



        contours= cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        if len(contours[0])==1:

            is_dilation_complete = True



        kernel_factor+=1

    

    #Draw the contours so it fills potential holes in the masks

    return cv2.drawContours(dilation, contours[0], 0, (255 , 255 , 255),thickness=cv2.FILLED)
def get_mask(img, x, y, width, height):



    #load the cropped area and apply an Otsu threshold

    cropped_img = np.array(img[y:y+height,x:x+width,:])

    blurred_img = cv2.GaussianBlur(cropped_img,(5,5),0)

    img_gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    ret, otsu = cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    

    #Place back the cropped area into a mask with the original image size

    img_height, img_width = img.shape[:2]

    img_mask = np.full((img_height,img_width),0)

    img_mask[y:y+height,x:x+width] = otsu



    return img_mask
def rle_encoding(x):

    dots = np.where(x.T.flatten() == 255)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return ' '.join([str(x) for x in run_lengths])
for i in range(5):

    sample_row = training_df.dropna().sample()

    random_image = sample_row.image_id.values[0]

    symbol_metadata_list = np.array(sample_row["labels"].values[0].split(' ')).reshape(-1, 5)

    

    x, y, width, height  = [int(x) for x in symbol_metadata_list[12][1:]]

    

    print("Creating mask for a random character in image {}".format(random_image))

    

    image_filename = "../../input/kuzushiji-recognition/train_images/{}.jpg".format(random_image)

    img = cv2.imread(image_filename,cv2.IMREAD_COLOR)

    mask = get_mask(img, x, y, width, height)



    cropped_img = img[y:y+height,x:x+width,:]

    cropped_mask = mask[y:y+height,x:x+width]

    

    cropped_single_mask = get_unique_mask(cropped_mask)

    

    #just organising everything to display

    cropped_mask = cropped_mask.copy()

    masked_img = np.zeros_like(cropped_img)

    masked_img[cropped_single_mask == 255] = cropped_img[cropped_single_mask == 255]

    mask[y:y+height,x:x+width] = cropped_single_mask 

    

    #display all the steps

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

    ax1.imshow(cropped_img)

    ax4.imshow(cropped_mask)

    ax2.imshow(masked_img)

    ax5.imshow(cropped_single_mask )

    ax3.imshow(img)

    ax6.imshow(mask)

    

    fig.tight_layout()

    plt.show()
training_df.head()
has_rle_been_processed = "EncodedPixels" in training_df.columns



#formatting the array that were converted to string when saving to .csv

#also getting rid of images with labels

if has_rle_been_processed:

    training_df["EncodedPixels"] = [[str_rle.strip(" ") for str_rle in rle_list.replace("'","").strip("[]").split(',')] for rle_list in training_df["EncodedPixels"]]

    training_df["CategoryId"] = [[int(value) for value in str_rle.replace(" ","").strip("[]").split(',') if value.isdigit()] for str_rle in training_df["CategoryId"]]

    training_df = training_df.drop(columns=['Unnamed: 0'])

    training_df = training_df.dropna()

    training_df = training_df[training_df['CategoryId'].map(lambda category_list: len(category_list)) > 0]
EncodedPixels_list = []

CategoryId_list = []

Width_list = []

Height_list = []



#If you use the dataframe provided, it will skip this preprocessing step

if not has_rle_been_processed:

    with tqdm(total=len(list(training_df.iterrows()))) as pbar:

        for idx, document_metadata in training_df.iterrows(): 

            pbar.update(1)



            document_rle_encoding = []

            document_categories = []



            try:

                image_filename = "../../input/kuzushiji-recognition/train_images/{}.jpg".format(document_metadata["image_id"])

                img = cv2.imread(image_filename,cv2.IMREAD_COLOR)



                #just getting the width and height of the image for my new metadata dataframe

                img_width, img_height = Image.open(image_filename).size



                #format the labels information so each item of the list represents a kuzushiji symbol

                symbol_metadata_list = np.array(document_metadata["labels"].split(' ')).reshape(-1, 5)



                for symbol_metadata in symbol_metadata_list:



                    symbol_unicode = symbol_metadata[0]

                    if symbol_unicode in kuzushiji_to_detect:

                        symbol_category = category_dict[symbol_unicode]

                        document_categories.append(symbol_category)



                        x, y, width, height  = [int(x) for x in symbol_metadata[1:]]



                        mask = get_mask(img, x, y, width, height)

                        cropped_single_mask = get_unique_mask(mask)

                        str_rle = rle_encoding(cropped_single_mask)



                        document_rle_encoding.append(str_rle)



            except:

                print("This document had no labels.")



            EncodedPixels_list.append(document_rle_encoding)

            CategoryId_list.append(document_categories)

            Width_list.append(img_width)

            Height_list.append(img_height)



    training_df["EncodedPixels"] = EncodedPixels_list

    training_df["CategoryId"] = CategoryId_list

    training_df["Width"] = Width_list

    training_df["Height"] = Height_list  
training_df.head()
def resize_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)  

    return img
class KuzushijiDataset(utils.Dataset):



    def __init__(self, df):

        super().__init__(self)

        

        # Add classes

        for i, name in enumerate(kuzushiji_to_detect):

            self.add_class("kuzushiji", i+1, name)

        

        # Add images 

        for i, row in df.iterrows():

            self.add_image("kuzushiji", 

                           image_id=row.name, 

                           path='../../input/kuzushiji-recognition/train_images/'+str(row.image_id)+".jpg", 

                           labels=row['CategoryId'],

                           annotations=row['EncodedPixels'], 

                           height=row['Height'], width=row['Width'])



    def image_reference(self, image_id):

        info = self.image_info[image_id]

        return info['path'], [kuzushiji_to_detect[int(x)] for x in info['labels']]

        

    def load_image(self, image_id):

        return resize_image(self.image_info[image_id]['path'])

        



    def load_mask(self, image_id):

        info = self.image_info[image_id]

                

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE, len(info['annotations'])), dtype=np.uint8)

        labels = []

        for m, (annotation, label) in enumerate(zip(info['annotations'], info['labels'])):

            sub_mask = np.full(info['height']*info['width'], 0, dtype=np.uint8)

            annotation = [int(x) for x in annotation.split(' ')]

            

            for i, start_pixel in enumerate(annotation[::2]):

                sub_mask[start_pixel: start_pixel+annotation[2*i+1]] = 1



            sub_mask = sub_mask.reshape((info['height'], info['width']), order='F')

            sub_mask = cv2.resize(sub_mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

            

            mask[:, :, m] = sub_mask

            labels.append(int(label)+1)

            

        return mask, np.array(labels)
training_percentage = 0.9



training_set_size = int(training_percentage*len(training_df))

validation_set_size = int((1-training_percentage)*len(training_df))



train_dataset = KuzushijiDataset(training_df[:training_set_size])

train_dataset.prepare()



valid_dataset = KuzushijiDataset(training_df[training_set_size:training_set_size+validation_set_size])

valid_dataset.prepare()



for i in range(5):

    image_id = random.choice(train_dataset.image_ids)

    print(train_dataset.image_reference(image_id))

    

    image = train_dataset.load_image(image_id)

    mask, class_ids = train_dataset.load_mask(image_id)

    visualize.display_top_masks(image, mask, class_ids, train_dataset.class_names, limit=5)
LR = 1e-4

EPOCHS = [1, 2]



import warnings 

warnings.filterwarnings("ignore")
model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)



#Using some pretrained weights from a previous kernel, otherwise we can use the COCO weights.

#We exclude some of the layers as I have just added more classes/symbols.

model.load_weights("../../input/maskrcnn-kuzushiji-pretrained-weights-v4/maskuzushiji_pretrained_weights_after_8_epochs.h5", by_name=True)

# model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[

#     'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
%%time

model.train(train_dataset, valid_dataset,

            learning_rate=LR,

            epochs=EPOCHS[0],

            layers='heads',

            augmentation=None)



history = model.keras_model.history.history
%%time

model.train(train_dataset, valid_dataset,

            learning_rate=LR,

            epochs=EPOCHS[1],

            layers='all')



new_history = model.keras_model.history.history

for k in new_history: history[k] = history[k] + new_history[k]
epochs = range(EPOCHS[-1])



plt.figure(figsize=(18, 6))



plt.subplot(131)

plt.plot(epochs, history['loss'], label="train loss")

plt.plot(epochs, history['val_loss'], label="valid loss")

plt.legend()

plt.subplot(132)

plt.plot(epochs, history['mrcnn_class_loss'], label="train class loss")

plt.plot(epochs, history['val_mrcnn_class_loss'], label="valid class loss")

plt.legend()

plt.subplot(133)

plt.plot(epochs, history['mrcnn_mask_loss'], label="train mask loss")

plt.plot(epochs, history['val_mrcnn_mask_loss'], label="valid mask loss")

plt.legend()



plt.show()
best_epoch = np.argmin(history["val_loss"]) + 1

print("Best epoch: ", best_epoch)

print("Valid loss: ", history["val_loss"][best_epoch-1])
class InferenceConfig(KuzushijiConfig):

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1



inference_config = InferenceConfig()



model = modellib.MaskRCNN(mode='inference', 

                          config=inference_config,

                          model_dir=ROOT_DIR)

glob_list = glob.glob(f'../../working/kuzushiji*/mask_rcnn_kuzushiji_{best_epoch:04d}.h5')

model_path = glob_list[0] if glob_list else ''

model.load_weights(model_path, by_name=True)
# Fix overlapping masks

def refine_masks(masks, rois):

    areas = np.sum(masks.reshape(-1, masks.shape[-1]), axis=0)

    mask_index = np.argsort(areas)

    union_mask = np.zeros(masks.shape[:-1], dtype=bool)

    for m in mask_index:

        masks[:, :, m] = np.logical_and(masks[:, :, m], np.logical_not(union_mask))

        union_mask = np.logical_or(masks[:, :, m], union_mask)

    for m in range(masks.shape[-1]):

        mask_pos = np.where(masks[:, :, m]==True)

        if np.any(mask_pos):

            y1, x1 = np.min(mask_pos, axis=1)

            y2, x2 = np.max(mask_pos, axis=1)

            rois[m, :] = [y1, x1, y2, x2]

    return masks, rois
sample_df = pd.read_csv("../../input/kuzushiji-recognition/sample_submission.csv")

sample_df.head()
for i in range(6):

    image_id = sample_df.sample()["image_id"].values[0]

    image_path = str('../../input/kuzushiji-recognition/test_images/'+image_id+'.jpg')

    print(image_path)

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    result = model.detect([resize_image(image_path)])

    r = result[0]

    

    if r['masks'].size > 0:

        masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)

        for m in range(r['masks'].shape[-1]):

            masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 

                                        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        

        y_scale = img.shape[0]/IMAGE_SIZE

        x_scale = img.shape[1]/IMAGE_SIZE

        rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)

        

        masks, rois = refine_masks(masks, rois)

    else:

        masks, rois = r['masks'], r['rois']

        

    visualize.display_instances(img, rois, masks, r['class_ids'], 

                                ['bg']+kuzushiji_to_detect, r['scores'],

                                title=image_id, figsize=(12, 12))
predictions = []

with tqdm(total=len(sample_df)) as pbar:

    for i,row in sample_df.iterrows():

        pbar.update(1)

        image_id = row["image_id"]

        image_path = str('../../input/kuzushiji-recognition/test_images/'+image_id+'.jpg')

        img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        result = model.detect([resize_image(image_path)])

        r = result[0]



        pred = ""

        if r['masks'].size > 0:

            masks = np.zeros((img.shape[0], img.shape[1], r['masks'].shape[-1]), dtype=np.uint8)

            for m in range(r['masks'].shape[-1]):

                masks[:, :, m] = cv2.resize(r['masks'][:, :, m].astype('uint8'), 

                                            (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)



            y_scale = img.shape[0]/IMAGE_SIZE

            x_scale = img.shape[1]/IMAGE_SIZE

            rois = (r['rois'] * [y_scale, x_scale, y_scale, x_scale]).astype(int)



            masks, rois = refine_masks(masks, rois)



            pred = " ".join(["{} {} {}".format(kuzushiji_to_detect[class_id],  int((roi[1]+(roi[3]-roi[1])/2)*x_scale), int((roi[0]+(roi[2]-roi[0])/2)*y_scale) ) for class_id, roi in zip(r['class_ids'],r['rois'])] )

        else:

            masks, rois = r['masks'], r['rois']



        predictions.append(pred)
sample_df["labels"] = predictions

sample_df.to_csv("../submission.csv",index=False)