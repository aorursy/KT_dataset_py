# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import matplotlib.patches as patches

from PIL import Image



# opencv for image analysis

import cv2

import urllib
from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

from bokeh.models import ColumnDataSource, HoverTool, Panel

from bokeh.models.widgets import Tabs
!cp -r "/kaggle/input/kerasretinanet/keras-retinanet" .
# install retinanet

!git clone https://github.com/fizyr/keras-retinanet.git

%cd keras-retinanet/



!pip install .



!python setup.py build_ext --inplace
import tensorflow as tf

from keras_retinanet import models

from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

from keras_retinanet.utils.visualization import draw_box, draw_caption

from keras_retinanet.utils.colors import label_color



import urllib

from tqdm.notebook import tqdm
train = pd.read_csv("/kaggle/input/global-wheat-detection/train.csv")  

train_data_folder = "/kaggle/input/global-wheat-detection/train/"

test_data_folder = "/kaggle/input/global-wheat-detection/test/"

train.head()
print("Dataset shape: {}".format(train.shape))
# check for missing values

train.isnull().sum()
train['width'].value_counts()
train['height'].value_counts()
# take random index

seed = 42

rng = np.random.RandomState(seed)
def show_images(df):

    # print a few images together with box

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), sharex=True, sharey=True)



    for r in range(nrows):

        for c in range(ncols):

            ridx = rng.choice(range(df.shape[0]))

            img_name = df.iloc[ridx]['image_id']



            image = plt.imread(train_data_folder+img_name+'.jpg')



            axs[r, c].imshow(image)

            axs[r, c].axis('off')

            

    plt.suptitle('Wheat head images')

    plt.show() 
# check a few random image to check the dataset.

show_images(train)
df_analysis=pd.DataFrame()

df_analysis['image_id']=train['image_id'].apply(lambda x: x+'.jpg')



# extract the fields for use

bbox = train.bbox.str.split(",",expand=True)

df_analysis['xmin'] = bbox[0].str.strip('[ ').astype(float)

df_analysis['ymin'] = bbox[1].str.strip(' ').astype(float)

df_analysis['xmax'] = bbox[2].str.strip(' ').astype(float)+df_analysis['xmin']

df_analysis['ymax'] = bbox[3].str.strip(' ]').astype(float)+df_analysis['ymin']

df_analysis['class']= 0



# show the data frame

df_analysis.head()
# take random index

# draw box on a random image

seed = 42

rng = np.random.RandomState(seed)
def show_images_with_box(df):

    # print a few images together with box

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15), sharex=True, sharey=True)



    for r in range(nrows):

        for c in range(ncols):

            ridx = rng.choice(range(df.shape[0]))

            img_name = df.iloc[ridx]['image_id']

      

            image = plt.imread(train_data_folder+img_name)

                        

            # find all the records of the provided image and draw box on the wheat heads

            chosen_image = df.loc[df["image_id"]==img_name,["xmin","ymin","xmax","ymax"]]

            bbox_array   = np.array(chosen_image.values.tolist())



            for bbox in bbox_array:

                image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color = (255,255,255), thickness=3) 



            axs[r, c].imshow(image)

            axs[r, c].axis('off')

            

    plt.suptitle('Images with Box')

    plt.show() 
show_images_with_box(df_analysis)
# lets plot images with high box counts (I am not stiching the boxes, as displaying them takes lot of time)

def show_images(se, start_idx, str_plot):

    # print a few images from the provided Series

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    idx = start_idx

    for r in range(nrows):

        for c in range(ncols):

            img = Image.open(train_data_folder+se.index[idx])

            axs[r, c].imshow(img)

            axs[r, c].axis('off')

            title="box_count="+str(se.iloc[idx])

            axs[r, c].set_title(title)

            

            idx = idx+1

    plt.suptitle(str_plot)
# lets see how the image box counts are ditributed.

img_freq=df_analysis['image_id'].value_counts()

img_freq[:10]
n, bins, patches = plt.hist(x=img_freq, bins='auto', color='#0504aa',

                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Box Count')

plt.ylabel('Frequency')

plt.title('Image Box Distribution')

plt.text(23, 45, r'$\mu=15, b=3$')

maxfreq = n.max()

# Set a clean upper y-axis limit.

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
import seaborn as sns



sns.set_style('darkgrid')

sns.distplot(img_freq)
# lets plot images with high box counts (I am not drawing the boxes, as displaying them takes some time)

def show_images(se, start_idx, str_plot):

    # print a few images from the provided Series

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    idx = start_idx

    for r in range(nrows):

        for c in range(ncols):

            img = Image.open(train_data_folder+se.index[idx])

            axs[r, c].imshow(img)

            axs[r, c].axis('off')

            title="box_count="+str(se.iloc[idx])

            axs[r, c].set_title(title)

            

            idx = idx+1

    plt.suptitle(str_plot)

    
show_images(img_freq, start_idx=0, str_plot='Images with max box counts')
show_images(img_freq, start_idx=(len(img_freq)-1-16), str_plot='Images with lowest box counts')
box_width=(df_analysis['xmax']-df_analysis['xmin']).sort_values()

box_height=(df_analysis['ymax']-df_analysis['ymin']).sort_values()
box_width
# plot box width

n, bins, patch = plt.hist(x=box_width, bins=50, color='#0504aa',

                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Box Width')

plt.ylabel('Frequency')

plt.title('Box Width Distribution')

plt.text(23, 45, r'$\mu=15, b=3$')

maxfreq = n.max()

# Set a clean upper y-axis limit.

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
sns.set_style('darkgrid')

sns.distplot(box_width)
# plot box height

n, bins, patch = plt.hist(x=box_height, bins=50, color='#0504aa',

                            alpha=0.7, rwidth=0.85)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('Box Height')

plt.ylabel('Frequency')

plt.title('Box Height Distribution')

plt.text(23, 45, r'$\mu=15, b=3$')

maxfreq = n.max()

# Set a clean upper y-axis limit.

plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
sns.set_style('darkgrid')

sns.distplot(box_height)
df_analysis['width']=df_analysis['xmax']-df_analysis['xmin']

df_analysis['height']=df_analysis['ymax']-df_analysis['ymin']



df_analysis.sort_values(by='width', inplace=True)

df_analysis.reset_index(inplace = True, drop = True)
df_analysis.head()
df_analysis.tail()
# function to print images with highest box width size. 

def show_box_width(df, start_idx, str_plot):

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    idx = start_idx

    for r in range(nrows):

        for c in range(ncols):

            img = plt.imread(train_data_folder+df.iloc[idx].image_id)

            axs[r, c].imshow(img)

            axs[r, c].axis('off')

            title="box width="+str(df.iloc[idx]['width'])

            axs[r, c].set_title(title)

                        

            # find all the records of the provided image and draw box on the wheat heads

            w, h, w1, h1 = df.iloc[idx][["xmin","ymin","xmax","ymax"]]

         

            img = cv2.rectangle(img, (int(w), int(h)), (int(w1), int(h1)), color = (255,255,255), thickness=3) 



            axs[r, c].imshow(img)

            axs[r, c].axis('off')

            idx = idx+1

    plt.suptitle(str_plot)
show_box_width(df_analysis, len(box_height)-16, 'Images with highest box width size')
show_box_width(df_analysis, 0, 'Images with lowest box width size')
# get image brightness using opencv

def get_image_brightness(img):

    image = cv2.imread(img)

    # convert to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    

    # get average brightness

    return int(np.array(gray).mean())
# create a data frame with unique image names

df_img=pd.DataFrame()



df_img['image']=df_analysis['image_id'].unique()
df_img['brightness'] = df_img['image'].apply(lambda x: get_image_brightness(train_data_folder+x))
df_img.sort_values(by='brightness', inplace=True)

df_img.reset_index(inplace = True, drop = True)

df_img.head()
df_img.tail()
# function to print images with highest box width size. 

def show_image_brightness(df, start_idx, str_plot):

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    idx = start_idx

    for r in range(nrows):

        for c in range(ncols):

            img = Image.open(train_data_folder+df.iloc[idx]['image'])

            axs[r, c].imshow(img)

            title="brightness="+str(df.iloc[idx]['brightness'])

            axs[r, c].set_title(title)



            axs[r, c].axis('off')

            idx = idx+1

    plt.suptitle(str_plot)
show_image_brightness(df_img, df_img.shape[0]-16,'most brightest images')
show_image_brightness(df_img, 0,'most darkest images')
def get_percentage_of_green_pixels(img):

    image = cv2.imread(img)

    # convert to HSV

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    

    # get the green mask

    hsv_lower = (40, 40, 40) 

    hsv_higher = (70, 255, 255)

    green_mask = cv2.inRange(hsv, hsv_lower, hsv_higher)

    

    return round(float(np.sum(green_mask)) / 255 / (1024 * 1024), 2)
# function to show images with highest box width size. 

def show_images_green_pixels(df, start_idx, str_plot):

    nrows=4

    ncols=4

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 15))

    idx = start_idx

    for r in range(nrows):

        for c in range(ncols):

            img = Image.open(train_data_folder+df.iloc[idx]['image'])

            axs[r, c].imshow(img)

            axs[r, c].axis('off')

            title="green pix="+str(df.iloc[idx]['green'])

            axs[r, c].set_title(title)



            axs[r, c].axis('off')

            idx = idx+1

    plt.suptitle(str_plot)
df_img['green'] = df_img['image'].apply(lambda x: get_percentage_of_green_pixels(train_data_folder+x))
df_img.sort_values(by='green', inplace=True)

df_img.reset_index(inplace = True, drop = True)

df_img.head()
df_img.tail()
sns.set_style('darkgrid')

sns.distplot(df_img['green'])
# display the greenest images

show_images_green_pixels(df_img, df_img.shape[0]-16,'most green images')
# display the less green(yellow) images

show_images_green_pixels(df_img, 0,'most yellow images')
test_data=[]

for root, dirs, files in os.walk(test_data_folder):

    for file in files:

        test_data.append(file)
print('test data count: {}'.format(len(test_data)))
# display test data

def show_test_data(tdata):

    nrows=2

    ncols=5

    fig, axs = plt.subplots(nrows, ncols, figsize=(20, 8), squeeze=False)

    idx = 0

    

    for r in range(nrows):

        for c in range(ncols):

            img = Image.open(test_data_folder+tdata[idx])

            axs[r, c].imshow(img)

            

            axs[r, c].axis('off')

            

            idx = idx+1

            

    plt.suptitle("Test data")
show_test_data(test_data)
# check the GPU details

!nvidia-smi
# creating a model using retinanet

#!keras_retinanet/bin/train.py --random-transform --gpu 0 --weights {PRETRAINED_MODEL} --lr {LR} --batch-size {BATCH_SIZE} --steps {STEPS} --epochs {EPOCHS} csv out_final.csv classes.csv

# this will create resnet50_csv_02.h5 as an output model.
# using a pre trained retinanet model first to try

model = models.load_model("/kaggle/input/retinanet-model1/resnet50_csv_02.h5", backbone_name='resnet50')

model = models.convert_model(model)
def predict(image):

    image = preprocess_image(image.copy())

    #image, scale = resize_image(image)



    boxes, scores, labels = model.predict_on_batch(

    np.expand_dims(image, axis=0)

  )



    #boxes /= scale



    return boxes, scores, labels
THRES_SCORE = 0.55



def draw_detections(image, boxes, scores, labels):

    for box, score, label in zip(boxes[0], scores[0], labels[0]):

        if score < THRES_SCORE:

            break



        color = label_color(label)



        b = box.astype(int)

        draw_box(image, b, color=color)



        caption = "{:.3f}".format(score)

        draw_caption(image, b, caption)
def show_detected_objects(image_name):

    img_path = test_data_folder+'/'+image_name

  

    image = read_image_bgr(img_path)

    #image = cv2.imread(img_path)



    boxes, scores, labels = predict(image)

    

    draw = image.copy()

    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



    draw_detections(draw, boxes, scores, labels)

    plt.figure(figsize=(8,6))

    plt.axis('off')

    plt.imshow(draw)

    plt.show()
for img in test_data:

    show_detected_objects(img)
preds=[]

imgid=[]

for img in test_data:

    img_path = test_data_folder+'/'+img

    image = read_image_bgr(img_path)

    boxes, scores, labels = predict(image)

    boxes=boxes[0]

    scores=scores[0]

    for idx in range(boxes.shape[0]):

        if scores[idx]>THRES_SCORE:

            box,score=boxes[idx],scores[idx]

            imgid.append(img.split(".")[0])

            preds.append("{} {} {} {} {}".format(score, int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])))

    
sub={"image_id":imgid, "PredictionString":preds}

sub=pd.DataFrame(sub)

sub.head()
sub_=sub.groupby(["image_id"])['PredictionString'].apply(lambda x: ' '.join(x)).reset_index()

sub_
samsub=pd.read_csv("/kaggle/input/global-wheat-detection/sample_submission.csv")

samsub.head()
for idx,imgid in enumerate(samsub['image_id']):

    samsub.iloc[idx,1]=sub_[sub_['image_id']==imgid].values[0,1]

    

samsub.head()
samsub.to_csv('/kaggle/working/submission.csv',index=False)