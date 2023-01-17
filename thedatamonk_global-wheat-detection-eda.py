import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.patches as patches



from PIL import Image



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/global-wheat-detection/train.csv")

sample_df = pd.read_csv("../input/global-wheat-detection/sample_submission.csv")



train_dir = "../input/global-wheat-detection/train/"

test_dir = "../input/global-wheat-detection/test/"
all_train_imgs = os.listdir(train_dir)

all_test_imgs = os.listdir(test_dir)



print ("Number of train images: {}".format(len(all_train_imgs)))

print ("Number of test images: {}".format(len(all_test_imgs)))



images_with_bbox = train_df.image_id.unique().tolist()

print ("Number of train images with bounding boxes: {}".format(len(images_with_bbox)))



images_without_bbox = []



for img_name in all_train_imgs:

    img_id = img_name[:-4]

    if img_id not in images_with_bbox:

        images_without_bbox.append(img_id)



print ("Number of train images without bounding boxes: {}".format(len(images_without_bbox)))
# Parse bounding box values

all_images_df = pd.DataFrame([image_name[:-4] for image_name in all_train_imgs], columns=['image_id'])



# now we will do a left join; images with no bounding boxes will have NaN values in all columns except `image_id`

all_images_df = pd.merge(all_images_df, train_df, how='left', on='image_id')



all_images_df.head()



# replace NaN values in width and height column

all_images_df.width.fillna(1024, inplace=True)

all_images_df.height.fillna(1024, inplace=True)



# replace NaN values in bbox column with [0, 0, 0, 0]

all_images_df.bbox.fillna('[0, 0, 0, 0]', inplace=True)



# parsing bbox column into 4 separate columns

bbox_info = all_images_df.bbox.str.split(', ', expand=True)



all_images_df['bbox_xmin'] = bbox_info[0].str.strip('[').astype('float')

all_images_df['bbox_ymin'] = bbox_info[1].str.strip(' ').astype('float')

all_images_df['bbox_width'] = bbox_info[2].str.strip(' ').astype('float')

all_images_df['bbox_height'] = bbox_info[3].str.strip(']').astype('float')





# dropping the source column as it not useful

all_images_df.drop(columns=['source'], inplace=True)



all_images_df.info()
# function to plot the bounding boxes for the training images



def return_all_bboxes(df, image_id):

    # select all rows with this image_id

    bboxes = df.loc[df['image_id'] == image_id, ['image_id', 'bbox_xmin', 'bbox_ymin', 'bbox_width', 'bbox_height']]

    return bboxes



def plot_images_with_bbox(df, rows = 3, cols = 3, fig_title="Training examples with bounding boxes"):

    # choose randomnly images from the training set

    unique_image_ids = df['image_id'].unique()

    image_ids = np.random.choice(unique_image_ids, rows*cols, replace=False)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    

    fig.suptitle(fig_title, fontsize=17)

    for ix in range(rows):

        for jx in range(cols):

            image_id = image_ids[ix * rows + jx]

            # plot the image

            img = Image.open(train_dir + image_id + ".jpg")

            axes[ix, jx].imshow(img)

            

            bboxes = return_all_bboxes(df, image_id)

            # plot the bounding box

            for _, bbox in bboxes.iterrows():

                rect = patches.Rectangle((bbox['bbox_xmin'], bbox['bbox_ymin']), bbox['bbox_width'], bbox['bbox_height'], linewidth=1, edgecolor='r', fill=False)

                axes[ix, jx].add_patch(rect)

            

            axes[ix, jx].set_axis_off()

plot_images_with_bbox(all_images_df)
no_of_bboxes = []

for image_id in images_with_bbox:

    bboxes = return_all_bboxes(all_images_df, image_id)

    no_of_bboxes.append(bboxes.shape[0])



# print (no_of_bboxes)
bbox_pvt_table = all_images_df.pivot_table(index=['image_id'], aggfunc='size')

bboxes_per_image_df = pd.DataFrame({'bboxes': bbox_pvt_table})

bboxes_per_image_df.reset_index(level=0, inplace=True)

bboxes_per_image_df.info()


sns.distplot(bboxes_per_image_df['bboxes'], bins=30, kde=False, hist_kws={'rwidth':0.75}, axlabel="# of bboxes / image")
lower_bound = 10

upper_bound = 90



image_ids_few_boxes = bboxes_per_image_df.loc[bboxes_per_image_df['bboxes'] <= lower_bound, 'image_id']

image_ids_many_boxes = bboxes_per_image_df.loc[bboxes_per_image_df['bboxes'] >= upper_bound, 'image_id']





print ("{} images have less than {} wheat heads in them".format(len(image_ids_few_boxes), lower_bound))

print ("{} images have more than {} wheat heads in them".format(len(image_ids_many_boxes), upper_bound))
few_boxes_df = all_images_df.loc[all_images_df['image_id'].isin(image_ids_few_boxes.values)]

plot_images_with_bbox(few_boxes_df, fig_title="Examples with very few boxes")
many_boxes_df = all_images_df.loc[all_images_df['image_id'].isin(image_ids_many_boxes.values)]

plot_images_with_bbox(many_boxes_df, fig_title="Examples with many boxes")
all_images_df['bbox_area'] = all_images_df['bbox_width'] * all_images_df['bbox_height']
print ("Largest area of a bounding box is: {}".format(all_images_df['bbox_area'].max()))
## We will find images with bounding boxes having area > 52000



area_threshold = 52000

very_large_bboxes_df = all_images_df.loc[all_images_df['bbox_area'] >= area_threshold]

print ("{} images have bounding boxes with an area > {}".format(very_large_bboxes_df.shape[0], area_threshold))

plot_images_with_bbox(very_large_bboxes_df, fig_title="Images with very large bounding boxes")