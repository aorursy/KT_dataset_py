import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import cv2

plt.style.use("seaborn")



import matplotlib.patches as patches

from glob import glob

from PIL import Image
#check the contents of the main directory

!ls ../input/global-wheat-detection
#check the count of images in the train direcory

!ls ../input/global-wheat-detection/train | wc -l
#before we start reading the data, create path variables for convience

folder_path = '../input/global-wheat-detection/'

TRAIN_IMAGES_PATH = folder_path + 'train/'

TEST_IMAGES_PATH = folder_path + 'test/'

TRAIN_CSV = folder_path + 'train.csv'
#read the data

train_df = pd.read_csv(TRAIN_CSV)
train_df.head()
#get the shape of the dataset

train_df.shape
#count the number of images in each directory using Glob function



train_glob = glob(TRAIN_IMAGES_PATH + '*')

test_glob = glob(TEST_IMAGES_PATH + '*')



print("Number of images in the train directory is {}".format(len(train_glob)))

print("Number of images in the test directory is {}".format(len(test_glob)))
#check if all the images have bounding boxes or not. Check the unique number of images in the train data with bounding boxes.



unique_count = len(train_df["image_id"].unique())

print("Number of unique images in the train dataset: {}".format(unique_count))

print("Number of images without bounding boxes is: {}".format(len(train_glob) - unique_count)) 
#validate the size of the image. check width and height is equal to 1024



(train_df["width"] == train_df["height"]).all()
#check the number of sources in the train data



len(train_df["source"].unique())
train_df["source"].value_counts(normalize = True).plot(kind = "barh")

plt.title("Distribution of images from different sources")

plt.xlabel("Percentage")

plt.show()
#Number of bounding box for each image - check the value counts



train_df["image_id"].value_counts().nlargest(5)
#create a new dataframe to store bounding box info

train_bbox_df = train_df[["image_id"]]

train_bbox_df["source"] = train_df["source"]
def extract_bbox(bbox_data):

    """Extract bbox data"""

    

    bbox_data = bbox_data.strip("[").strip("]").split(",")

    bbox_xmin = float(bbox_data[0])

    bbox_ymin = float(bbox_data[1])

    bbox_xmax = float(bbox_data[0]) + float(bbox_data[2])

    bbox_ymax = float(bbox_data[1]) + float(bbox_data[3])

    bbox_w = float(bbox_data[2])

    bbox_h = float(bbox_data[3])

    

    return bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, bbox_w, bbox_h
#extract the bounding box data

train_bbox_df["bbox_xmin"],train_bbox_df["bbox_ymin"], train_bbox_df["bbox_xmax"], train_bbox_df["bbox_ymax"],train_bbox_df["bbox_w"], train_bbox_df["bbox_h"] = zip(*train_df["bbox"].map(extract_bbox))
#function to display the images



def get_all_bboxes(df, image_id):

    image_bboxes = df[df.image_id == image_id]

    

    bboxes = []

    for _,row in image_bboxes.iterrows():

        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_w, row.bbox_h))

    return bboxes



def plot_image_samples(df, rows=3, cols=3, title='Image examples', bln_bbox = True, bln_save = False):

    fig, axs = plt.subplots(rows, cols, figsize=(10,10))

    for row in range(rows):

        for col in range(cols):

            idx = np.random.randint(len(df), size=1)[0]

            img_id = df.iloc[idx].image_id

            img = Image.open(TRAIN_IMAGES_PATH + img_id + '.jpg')

            axs[row, col].imshow(img)

            

            if bln_bbox == True:                

                bboxes = get_all_bboxes(df, img_id)

                for bbox in bboxes:

                    rect = patches.Rectangle((bbox[0],bbox[1]),bbox[2],bbox[3],linewidth=1,edgecolor='r',facecolor='none')

                    axs[row, col].add_patch(rect)

            

            axs[row, col].axis('off')

            

    plt.suptitle(title)

    if bln_save == True:

        plt.savefig('sample.png', dpi=200)

    
#compute the bounding box area



train_bbox_df["area"] = train_bbox_df["bbox_w"] * train_bbox_df["bbox_h"]
train_bbox_df.head()
#plot images without bounding boxes



plot_image_samples(train_bbox_df, bln_bbox = False, rows = 3, cols = 3, title = "sample images")
#with bounding box

plot_image_samples(train_bbox_df, bln_bbox = True, rows = 3, cols = 3, title = "Image examples with bounding boxes", bln_save = True)
#plot images without bounding boxes

plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "usask_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `usask_1'")
#plot images without bounding boxes

plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "arvalis_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `arvalis_1'")
#plot images without bounding boxes

plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "inrae_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `inrae_1'")
#plot images without bounding boxes

plot_image_samples(train_bbox_df.loc[train_bbox_df["source"] == "ethz_1"], bln_bbox = True, rows = 3, cols = 3, title = "Images from `ethz_1'")
train_bbox_df.area.value_counts().nlargest(5)
#basic stats of the area

train_bbox_df["area"].describe()
#boxplot to find out any large bounding boxes

fig, ax = plt.subplots(ncols= 2, figsize = (14,6))    



#boxplot for comparison

sns.boxplot(y = "area", data = train_bbox_df, ax=ax[0])

ax[0].set_title("Box plot of bounding box area to analyze abnormal sizes")



#distribution plot

ax[1].set_title("Distribution of bounding box area")

ax[1].set_ylabel("Frequency")

sns.distplot(a = train_bbox_df["area"], ax=ax[1], kde=False, bins = 150)



plt.show()
#from the boxplot we can see that they are 3 instances where the bounding box area is more than 300,000.

train_bbox_df.loc[train_bbox_df["area"] > 300000]
plot_image_samples(train_bbox_df.loc[train_bbox_df["area"] > 300000], title = "Images where bounding boxes area is more than 300,000")
# we will look at all the data in the outliers

plot_image_samples(train_bbox_df.loc[train_bbox_df["area"] > 100000], title = "Images where bounding boxes area is more than 100,000")
# we will look at all the data in the outliers

plot_image_samples(train_bbox_df.loc[train_bbox_df["area"] > 100000], title = "Images where bounding boxes area is more than 100,000")