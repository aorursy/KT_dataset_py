import numpy as np

import pandas as pd
training_file='/kaggle/input/global-wheat-detection/train.csv'

training_folder='/kaggle/input/global-wheat-detection/train/'



training_labels = pd.read_csv(training_file)

training_labels.head()
image_ids = training_labels.drop_duplicates(subset='image_id')

image_ids.head()
image_ids.groupby(['source']).size().plot(kind='bar')
image_ids.groupby(['width']).size().plot(kind='bar')
image_ids.groupby(['height']).size().plot(kind='bar')
images_with_wheat = image_ids['image_id'].count()
import glob

training_folder_image_list = [f for f in glob.glob(training_folder+"*.jpg")]

total_images = len(training_folder_image_list)
percent = images_with_wheat / total_images * 100

print("Percentage of images with wheat: {}".format(percent))
training_labels['image_id'].value_counts()
training_labels['image_id'].value_counts().plot(kind='line')
arvalis_1 = training_labels[training_labels['source'] == 'arvalis_1']

arvalis_2 = training_labels[training_labels['source'] == 'arvalis_2']

arvalis_3 = training_labels[training_labels['source'] == 'arvalis_3']

ethz_1 = training_labels[training_labels['source'] == 'ethz_1']

inrae_1 = training_labels[training_labels['source'] == 'inrae_1']

rres_1 = training_labels[training_labels['source'] == 'rres_1']

usask_1 = training_labels[training_labels['source'] == 'usask_1']
import matplotlib.pyplot as plt

fig1 = plt.figure()

ax1 = fig1.add_subplot(111)



ax1.boxplot([arvalis_1['image_id'].value_counts(),

             arvalis_2['image_id'].value_counts(),

             arvalis_3['image_id'].value_counts(),

             ethz_1['image_id'].value_counts(),

             inrae_1['image_id'].value_counts(),

             usask_1['image_id'].value_counts(),

             rres_1['image_id'].value_counts(),])

plt.show()
from IPython.display import Image

wheat_image = Image(training_folder+training_labels['image_id'][0]+".jpg")

wheat_image
wheat_image = Image(training_folder+training_labels['image_id'][200]+".jpg")

wheat_image
training_labels_full_path = [training_folder+image+".jpg" for image in training_labels["image_id"]]

non_wheat_images = []

for image in training_folder_image_list:

    if not image in training_labels_full_path:

        non_wheat_images.append(image)

non_wheat_image = Image(non_wheat_images[0])

non_wheat_image
non_wheat_image2 = Image(non_wheat_images[1])

non_wheat_image2
non_wheat_image3 = Image(non_wheat_images[30])

non_wheat_image3