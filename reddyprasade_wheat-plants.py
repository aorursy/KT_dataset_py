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

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import os

from PIL import Image, ImageDraw

from ast import literal_eval

import matplotlib.pyplot as plt

import cv2
root_path = "../input/global-wheat-detection/"

train_folder = os.path.join(root_path, "train")

test_folder = os.path.join(root_path, "test")

train_csv_path = os.path.join(root_path, "train.csv")
df = pd.read_csv(train_csv_path)
df.head()
df.shape[0]
df['width'].unique() == df['height'].unique() == [1024]
import pandas_profiling

import seaborn as sns
pandas_profiling.ProfileReport(df)
def get_bbox_area(bbox):

    bbox = literal_eval(bbox)

    return bbox[2] * bbox[3]
df['bbox_area'] = df['bbox'].apply(get_bbox_area)
df['bbox_area'].value_counts().hist(bins=50)
# checking missing data

total = df.isnull().sum().sort_values(ascending = False)

percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train_data.head()
def plot_count(df, feature, title='', size=2):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))

    total = float(len(df))

    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')

    plt.title(title)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count(df=df, feature='source', title = 'data source count and %age plot', size=3)
unique_images = df['image_id'].unique()
num_total = len(os.listdir(train_folder))

num_annotated = len(unique_images)



print(f"There are {num_annotated} annotated images and {num_total - num_annotated} images without annotations.")
sources = df['source'].unique()

print(f"There are {len(sources)} sources of data: {sources}")
df['source'].value_counts()
plt.hist(df['image_id'].value_counts(), bins=30)

plt.show()
def show_images(images, num = 5):

    

    images_to_show = np.random.choice(images, num)



    for image_id in images_to_show:



        image_path = os.path.join(train_folder, image_id + ".jpg")

        image = Image.open(image_path)



        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = [literal_eval(box) for box in df[df['image_id'] == image_id]['bbox']]



        # visualize them

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:    

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=4)



        plt.figure(figsize = (15,15))

        plt.imshow(image)

        plt.show()
show_images(unique_images)
def display_images(images): 

    f, ax = plt.subplots(5,3, figsize=(18,22))

    for i, image_id in enumerate(images):

        image_path = os.path.join(train_folder, f'{image_id}.jpg')

        image = Image.open(image_path)

        

        # get all bboxes for given image in [xmin, ymin, width, height]

        bboxes = [literal_eval(box) for box in df[df['image_id'] == image_id]['bbox']]

        # draw rectangles on image

        draw = ImageDraw.Draw(image)

        for bbox in bboxes:    

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)

            

        ax[i//3, i%3].imshow(image) 

        image.close()       

        ax[i//3, i%3].axis('off')



        source = df[df['image_id'] == image_id]['source'].values[0]

        ax[i//3, i%3].set_title(f"image_id: {image_id}\nSource: {source}")

        plt.savefig("image_id.jpg",dip=150)



    plt.show() 
images = df.sample(n=15, random_state=42)['image_id'].values

display_images(images)
submission = pd.read_csv(f'{root_path}/sample_submission.csv')
# since we need to predict bounding boxes for test images, hence below images do not have any bounding boxes

test_images = submission.image_id.values
test_images
def display_test_images(images): 

    f, ax = plt.subplots(5,2, figsize=(20, 50))

    for i, image_id in enumerate(images):

        image_path = os.path.join(test_folder, f'{image_id}.jpg')

        image = Image.open(image_path)        

            

        ax[i//2, i%2].imshow(image) 

        ax[i//2, i%2].axis('off')

        ax[i//2, i%2].set_title(f"image_id: {image_id}")

        plt.savefig("Test_sample.png",dip=150)



    plt.show()
display_test_images(test_images)