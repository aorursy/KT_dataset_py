import numpy as np

import pandas as pd

import os

import imageio

import scipy.ndimage as ndi

import matplotlib.pyplot as plt

import cv2

from ast import literal_eval

from PIL import Image, ImageDraw
source_path = "../input/global-wheat-detection"



train_imgs_path = os.path.join(source_path, "train")

test_imgs_path = os.path.join(source_path, "test")



csv_train_data_path = "../input/global-wheat-detection/train.csv"



samp_subm = '../input/global-wheat-detection/sample_submission.csv'

print('Available images for training: ', len(os.listdir(train_imgs_path)))

print('Available images for testing: ', len(os.listdir(test_imgs_path)))
sizes = []

sizes.append(len(os.listdir(train_imgs_path)))

sizes.append(len(os.listdir(test_imgs_path)))



plt.figure(figsize= (10,10))

plt.pie(sizes, explode=(0,0), labels=['train_images', 'test_images'],autopct='%1.1f%%', shadow=False, startangle=45)



plt.title('Percentage of Wheat images available in Train and Test Sets',fontsize = 18)

plt.axis('equal')

plt.tight_layout()


csv_train_data = pd.read_csv(csv_train_data_path)



csv_train_data.head()

available_imgs = csv_train_data['image_id'].unique()

print('Available annotated images: ', len(available_imgs))

print('Available images without annotations: ', len(os.listdir(train_imgs_path))-len(available_imgs))
sizes = []

sizes.append(len(available_imgs))

sizes.append(len(os.listdir(train_imgs_path))-len(available_imgs))



plt.figure(figsize= (10,10))

plt.pie(sizes, explode=(0.2,0), labels=['annotation', '~annotation'],autopct='%1.1f%%', shadow=False, startangle=45)



plt.title('Annotation vs ~Annotation',fontsize = 18)

plt.axis('equal')

plt.tight_layout()
labels = ['ethz_1','arvalis_1','rres_1', 'arvalis_3', 'usask_1', 'arvalis_2', 'inrae_1', ]

sizes = csv_train_data['source'].value_counts()



explode = []



for i in labels:

    explode.append(0.05)

    

plt.figure(figsize= (10,10))

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode)

centre_circle = plt.Circle((0,0),0.70,fc='white')



fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.title('Visualization of Wheat images from different sources',fontsize = 20)

plt.axis('equal')  

plt.tight_layout()
plt.figure(figsize= (10,10))

y,x,_ = plt.hist(csv_train_data['image_id'].value_counts(), bins=100, density = True, color = 'red')



print('Minimum no. of Bounding Boxes: ', x.min())

print('Maximum no. of Bounding Boxes: ', x.max())



plt.xlabel('# Bboxes')

plt.ylabel('Probability')

plt.title('Histogram of describing probabilities of Bboxes')



plt.grid(True)



plt.show()
def view_imgs(imgs):

    imgs_to_view = np.random.choice(imgs, 10)

    for image_id in imgs_to_view:

        img_path = os.path.join(train_imgs_path, image_id + '.jpg')

        #img = imageio.imread(img_path)

        img = Image.open(img_path)

        bboxes = [literal_eval(box) for box in csv_train_data[csv_train_data['image_id']==image_id]['bbox']]

        draw = ImageDraw.Draw(img)

        for bbox in bboxes:

            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],outline = (255,0,0),width=2)

        plt.figure(figsize=(10,10))

        plt.imshow(img)

        plt.show()

        

    
view_imgs(available_imgs)