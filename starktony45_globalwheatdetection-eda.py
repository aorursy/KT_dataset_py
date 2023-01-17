# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from glob import glob

import os

from PIL import Image

from matplotlib import patches

from bokeh.models import ColumnDataSource, HoverTool, Panel

from bokeh.models.widgets import Tabs

from bokeh.plotting import figure

from bokeh.io import output_notebook, show, output_file

%matplotlib inline

import cv2
train_dir ='/kaggle/input/global-wheat-detection/train/'

test_dir = '../input/global-wheat-detection/test/'

train_csv_path = '../input/global-wheat-detection/train.csv' 
train =pd.read_csv(train_csv_path)

train.head()



train_images = glob(train_dir+ '*')

test_images = glob(test_dir + '*')

print("The images in train images are ",len(train_images))

print("The images in test images are ",len(test_images))
train.head()
#train_images

all_train_images = pd.DataFrame(i.split('/')[-1][:-4] for i in train_images)

all_train_images.columns = ['image_id']

all_train_images = all_train_images.merge(train,on = 'image_id',how='left')
all_train_images.head()
all_train_images['bbox'] = all_train_images['bbox'].fillna('[0,0,0,0]')

bbox_items = all_train_images['bbox'].str.split(',',expand = True)

all_train_images['bbox_xmin'] = bbox_items[0].str.strip('[').astype(float)

all_train_images['bbox_ymin'] = bbox_items[1].str.strip(' ').astype(float)

all_train_images['bbox_width'] = bbox_items[2].str.strip(' ').astype(float)

all_train_images['bbox_height'] = bbox_items[3].str.strip(']').astype(float)

all_train_images
print("Images without heads is",len(all_train_images)-len(train))
def get_all_boxes(df,image_id):

    bboxes = []

    image_bbox = df[df.image_id==image_id]

    for _,rows in image_bbox.iterrows():

        bboxes.append((rows.bbox_xmin,rows.bbox_ymin,rows.bbox_width,rows.bbox_height))

        

    return bboxes



def plot_image_examples(df,rows= 3,columns=3,title ='Image Examples'):

    fig,axs = plt.subplots(rows,columns,figsize=(10,10))

    for row in range(rows):

        for col in range(columns):

            idx = np.random.randint(len(df),size=1)[0]

            img_id = df.iloc[idx].image_id

            

            img = Image.open(train_dir + img_id + '.jpg')

            

            axs[row,col].imshow(img)

            

            bboxes = get_all_boxes(df,img_id)

            

            for bbox in bboxes:

                

                rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[2],edgecolor='r',linewidth=1,facecolor='none')

                axs[row,col].add_patch(rect)

                

            axs[row,col].axis('off')

            

    plt.suptitle(title)

            

plot_image_examples(all_train_images)
all_train_images['width'].value_counts()
all_train_images['counts'] = all_train_images.apply(lambda row: 1 if np.isfinite(row.width) else 0,axis =1)

train_images_count = all_train_images.groupby('image_id').sum().reset_index()
train_images_count
# See this article on how to plot bar charts with Bokeh:

# https://towardsdatascience.com/interactive-histograms-with-bokeh-202b522265f3

def hist_hover(dataframe, column, colors=["#94c8d8", "#ea5e51"], bins=30, title=''):

    hist, edges = np.histogram(dataframe[column], bins = bins)

    

    hist_df = pd.DataFrame({column: hist,

                             "left": edges[:-1],

                             "right": edges[1:]})

    hist_df["interval"] = ["%d to %d" % (left, right) for left, 

                           right in zip(hist_df["left"], hist_df["right"])]



    src = ColumnDataSource(hist_df)

    plot = figure(plot_height = 400, plot_width = 600,

          title = title,

          x_axis_label = column,

          y_axis_label = "Count")    

    plot.quad(bottom = 0, top = column,left = "left", 

        right = "right", source = src, fill_color = colors[0], 

        line_color = "#35838d", fill_alpha = 0.7,

        hover_fill_alpha = 0.7, hover_fill_color = colors[1])

        

    hover = HoverTool(tooltips = [('Interval', '@interval'),

                              ('Count', str("@" + column))])

    plot.add_tools(hover)

    

    output_notebook()

    show(plot)
hist_hover(train_images_count,'counts','Number of wheat spikes per image')
#Lets plot some image with less number of count

less_spikes = train_images_count[train_images_count['counts']<10].image_id
plot_image_examples(all_train_images[all_train_images.image_id.isin(less_spikes)],title = 'Images with less spikes')
#Plotting the images with highest spikes

more_spikes = train_images_count[train_images_count['counts']>100].image_id
plot_image_examples(all_train_images[all_train_images.image_id.isin(more_spikes)],title= 'High number of Spikes')
all_train_images['bbox_area'] = all_train_images['bbox_width']*all_train_images['bbox_height']
hist_hover(all_train_images,'bbox_area',title ='Area of a single bounding box')
#The max area of bounding box

max(all_train_images['bbox_area'])
large_area = all_train_images[all_train_images['bbox_area'] >200000].image_id
plot_image_examples(all_train_images[all_train_images.image_id.isin(large_area)],title = 'Large bbox area in a image')
small_area = all_train_images[all_train_images['bbox_area']<50].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(small_area)],title='Small bbox area in images')
area_per_image = all_train_images.groupby("image_id").sum().reset_index()
area_per_image_percentage = area_per_image.copy()

area_per_image_percentage['bbox_area'] = area_per_image['bbox_area']/(1024*1024)*100
area_per_image.head()
area_per_image_percentage.head()
hist_hover(area_per_image_percentage,'bbox_area',title ='Percentage of image covered by bbox')
small_percentage = area_per_image_percentage[area_per_image_percentage['bbox_area']<8].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(small_percentage)],title='low area covered by bbox')
high_percentage = area_per_image_percentage[area_per_image_percentage['bbox_area']>50].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(high_percentage)],title='high area covered by bbox')
def get_brightness(image):

    

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    

    return np.array(gray).mean()

    

def add_brightness(df):

    

    brightness = []

    for _, row in df.iterrows():

        img_id = row.image_id

        image = cv2.imread(train_dir+img_id+'.jpg')

        brightness.append(get_brightness(image))

        

    brightness_df = pd.DataFrame(brightness)

    brightness_df.columns = ['brightness']

    df = pd.concat([df,brightness_df],ignore_index = True,axis=1)

    df.columns = ['image_id','brightness']

    

    return df









    

image_df = pd.DataFrame(all_train_images.image_id.unique())
image_df.columns = ['image_id']
brightness_df = add_brightness(image_df)



all_train_images = all_train_images.merge(brightness_df,on='image_id')
hist_hover(all_train_images,'brightness',title ='Brightness in images')
dark_ids = all_train_images[all_train_images['brightness']<25].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(dark_ids)],title='The image with low brightness')
bright_ids = all_train_images[all_train_images['brightness']>130].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(bright_ids)],title='The image with high brightness')
def green_pixels(image):

    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    

    #Get the green mask. I got from "https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image"

    low =(40,40,40)

    high = (70,255,255)

    green_mask = cv2.inRange(img,low,high)

    

    return float( np.sum(green_mask))/255/(1024*1024)



def yellow_pixels(image):

    img = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    low= (25,40,40)

    high = (35,255,255)

    yellow_mask = cv2.inRange(img,low,high)

    

    return float(np.sum(yellow_mask))/255/(1024*1024)





def add_green(df):

    

    brightness = []

    for _, row in df.iterrows():

        img_id = row.image_id

        image = cv2.imread(train_dir+img_id+'.jpg')

        brightness.append(green_pixels(image))

        

    brightness_df = pd.DataFrame(brightness)

    brightness_df.columns = ['green_bright']

    df = pd.concat([df,brightness_df],ignore_index = True,axis=1)

    df.columns = ['image_id','green_bright']

    

    return df



def add_yellow(df):

    

    brightness = []

    for _, row in df.iterrows():

        img_id = row.image_id

        image = cv2.imread(train_dir+img_id+'.jpg')

        brightness.append(yellow_pixels(image))

        

    brightness_df = pd.DataFrame(brightness)

    brightness_df.columns = ['yellow_bright']

    df = pd.concat([df,brightness_df],ignore_index = True,axis=1)

    df.columns = ['image_id','yellow_bright']

    

    return df





    
green_pixels_df = add_green(image_df)

all_train_images = all_train_images.merge(green_pixels_df,on='image_id')
hist_hover(all_train_images,'green_bright',title ='Green Colors in images')
green_ids = all_train_images[all_train_images['green_bright']>0.4].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(green_ids)],title='The image with high green color')
yellow_pixels_df = add_yellow(image_df)

all_train_images = all_train_images.merge(yellow_pixels_df,on='image_id')
hist_hover(all_train_images,'yellow_bright',title ='yellow Colors in images')
yellow_ids = all_train_images[all_train_images['yellow_bright']>0.55].image_id

plot_image_examples(all_train_images[all_train_images.image_id.isin(yellow_ids)],title='The image with high yellow color')
import albumentations as al

example = al.Compose([

    al.RandomSizedBBoxSafeCrop(512,512,erosion_rate=0.0,interpolation=1,p=1.0),

    al.HorizontalFlip(p=0.5),

    al.VerticalFlip(p=0.5),

    al.OneOf([al.RandomContrast(),

             al.RandomGamma(),

             al.RandomBrightness()],p=1.0),

    al.CLAHE(p=0.1)], p=1.0, bbox_params=al.BboxParams(format='coco', label_fields=['category_id']))
def apply_transform(transforms,df,n_transforms=3):

    idx = np.random.randint(len(df),size=1)[0]

    bboxes = []

    image_id = df.iloc[idx].image_id

    image_bbox = df[df.image_id==image_id]

    for _,rows in image_bbox.iterrows():

        bboxes.append([rows.bbox_xmin,rows.bbox_ymin,rows.bbox_width,rows.bbox_height])

        

    





            

    img = Image.open(train_dir + image_id + '.jpg')

            

    fix,axs = plt.subplots(1,n_transforms+1,figsize=(15,7))

            

    axs[0].imshow(img)

    axs[0].set_title("Original")

            

    for bbox in bboxes:

        

        rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],edgecolor='r',linewidth=1,facecolor='none')

        axs[0].add_patch(rect)

                

    # apply transforms n_transforms times

    for i in range(n_transforms):

        params = {'image': np.asarray(img),

                  'bboxes': bboxes,

                  'category_id': [1 for j in range(len(bboxes))]}

        augmented_boxes = transforms(**params)

        bboxes_aug = augmented_boxes['bboxes']

        image_aug = augmented_boxes['image']



        # plot the augmented image and augmented bounding boxes

        axs[i+1].imshow(image_aug)

        axs[i+1].set_title('augmented_' + str(i+1))

        for bbox in bboxes_aug:

            rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

            axs[i+1].add_patch(rect)

    plt.show()

            

apply_transform(example,all_train_images,n_transforms=3)
apply_transform(example,all_train_images,n_transforms=3)