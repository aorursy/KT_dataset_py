import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import glob

import math

import random

import time

import datetime

from collections import defaultdict

from tqdm import tqdm, tqdm_notebook



import xml.etree.ElementTree as ET 



import cv2



print(os.listdir("../input"))
image_width = 64

image_height = 64

image_channels = 3

image_sample_size = 10000

image_output_dir = '../output_images/'

image_input_dir = '../input/all-dogs/all-dogs/'

image_ann_dir = "../input/annotation/Annotation/"
dog_breed_dict = {}

for annotation in os.listdir(image_ann_dir):

    annotations = annotation.split('-')

    dog_breed_dict[annotations[0]] = annotations[1]
print(dog_breed_dict['n02097658'])
def get_input_image_dict(image_input_dir, labels_dict):

    image_sample_dict = defaultdict(list)

    for image in os.listdir(image_input_dir):

        filename = image.split('.')

        label_code = filename[0].split('_')[0]

        breed_name = labels_dict[label_code]

        #print('Code: {}, Breed: {}'.format(label_code, breed_name))

        if image is not None:

            image_sample_dict[breed_name].append(image)

    

    print('Created label dictionary for input images.')

    return image_sample_dict
image_sample_dict = get_input_image_dict(image_input_dir, dog_breed_dict)
def plot_class_distributions(image_sample_dict, title=''):

    class_lengths = []

    labels = []

    total_images = 0

    

    print('Total amount of dog breeds: ', len(image_sample_dict))

    

    for label, _ in image_sample_dict.items():

        total_images += len(image_sample_dict[label])

        class_lengths.append(len(image_sample_dict[label]))

        labels.append(label)

        

    print('Total amount of input images: ', total_images)

        

    plt.figure(figsize = (10,30))

    plt.barh(range(len(class_lengths)), class_lengths)

    plt.yticks(range(len(labels)), labels)

    plt.title(title)

    plt.ylabel('Dog Breed')

    plt.xlabel('Sample size')

    plt.show()

    

    return total_images
total_images = plot_class_distributions(image_sample_dict)
def read_image(src):

    img = cv2.imread(src)

    if img is None:

        raise FileNotFoundError

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
def plot_images(directory, image_sample_dict, examples=25, disp_labels=True): 

  

    if not math.sqrt(examples).is_integer():

        print('Please select a valid number of examples.')

        return

    

    imgs = []

    classes = []

    for i in range(examples):

        rnd_class, _ = random.choice(list(image_sample_dict.items()))

        #print(rnd_class)

        rnd_idx = np.random.randint(0, len(image_sample_dict[rnd_class]))

        filename = image_sample_dict[rnd_class][rnd_idx]

        img = read_image(os.path.join(directory, filename))

        imgs.append(img)

        classes.append(rnd_class)

    

    

    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)),figsize=(15,15),

    subplot_kw = {'xticks':[], 'yticks':[]},

    gridspec_kw = dict(hspace=0.3, wspace=0.1))

    

    for i, ax in enumerate(axes.flat):

        if disp_labels == True:

            ax.title.set_text(classes[i])

        ax.imshow(imgs[i])
plot_images(image_input_dir, image_sample_dict)
plot_images(image_input_dir, image_sample_dict, examples=36, disp_labels=True)
def load_cropped_images(dog_breed_dict=dog_breed_dict, image_ann_dir=image_ann_dir, sample_size=25000, 

                        image_width=image_width, image_height=image_height, image_channels=image_channels):

    curIdx = 0

    breeds = []

    dog_images_np = np.zeros((sample_size,image_width,image_height,image_channels))

    for breed_folder in os.listdir(image_ann_dir):

        for dog_ann in tqdm(os.listdir(image_ann_dir + breed_folder)):

            try:

                img = read_image(os.path.join(image_input_dir, dog_ann + '.jpg'))

            except FileNotFoundError:

                continue

                

            tree = ET.parse(os.path.join(image_ann_dir + breed_folder, dog_ann))

            root = tree.getroot()

            

            size = root.find('size')

            width = int(size.find('width').text)

            height = int(size.find('height').text)

            objects = root.findall('object')

            for o in objects:

                bndbox = o.find('bndbox') 

                

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                

                xmin = max(0, xmin - 4)        # 4 : margin

                xmax = min(width, xmax + 4)

                ymin = max(0, ymin - 4)

                ymax = min(height, ymax + 4)



                w = np.min((xmax - xmin, ymax - ymin))

                w = min(w, width, height)                     # available w



                if w > xmax - xmin:

                    xmin = min(max(0, xmin - int((w - (xmax - xmin))/2)), width - w)

                    xmax = xmin + w

                if w > ymax - ymin:

                    ymin = min(max(0, ymin - int((w - (ymax - ymin))/2)), height - w)

                    ymax = ymin + w

                

                img_cropped = img[ymin:ymin+w, xmin:xmin+w, :]      # [h,w,c]

                # Interpolation method

                if xmax - xmin > image_width:

                    interpolation = cv2.INTER_AREA          # shrink

                else:

                    interpolation = cv2.INTER_CUBIC         # expansion

                    

                img_cropped = cv2.resize(img_cropped, (image_width, image_height), 

                                         interpolation=interpolation)  # resize

                    

                dog_images_np[curIdx,:,:,:] = np.asarray(img_cropped)

                dog_breed_name = dog_breed_dict[dog_ann.split('_')[0]]

                breeds.append(dog_breed_name)

                curIdx += 1

                

    dog_images_np = dog_images_np / 255.  # change the pixel range to [-1, 1]

    return dog_images_np, breeds
start_time = time.time()

dog_images_np, breeds = load_cropped_images()

est_time = round(time.time() - start_time)

print("Feature loading time: {}.".format(str(datetime.timedelta(seconds=est_time))))
def plot_features(features, labels, image_width=image_width, image_height=image_height, 

                image_channels=image_channels,

                examples=25, disp_labels=True): 

  

    if not math.sqrt(examples).is_integer():

        print('Please select a valid number of examples.')

        return

    

    imgs = []

    classes = []

    for i in range(examples):

        rnd_idx = np.random.randint(0, len(labels))

        imgs.append(features[rnd_idx, :, :, :])

        classes.append(labels[rnd_idx])

    

    

    fig, axes = plt.subplots(round(math.sqrt(examples)), round(math.sqrt(examples)),figsize=(15,15),

    subplot_kw = {'xticks':[], 'yticks':[]},

    gridspec_kw = dict(hspace=0.3, wspace=0.01))

    

    for i, ax in enumerate(axes.flat):

        if disp_labels == True:

            ax.title.set_text(classes[i])

        ax.imshow(imgs[i])
print('Loaded features shape: ', dog_images_np.shape)

print('Loaded labels: ', len(breeds))
print('Plotting cropped images by specified coordinates..')

plot_features(dog_images_np, breeds, examples=16, disp_labels=True)
plt.imshow(dog_images_np[3])