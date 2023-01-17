import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import cv2

import re



%matplotlib inline
df_train = pd.read_csv('../input/kuzushiji-recognition/train.csv')

unicode_map = {codepoint: char for codepoint, char in 

               pd.read_csv('../input/kuzushiji-recognition/unicode_translation.csv').values}
def convert_labels_set(labels_str):

    labels = []

    for one_label_str in re.findall(r'U\+\S+\s\S+\s\S+\s\S+\s\S+', labels_str):

        charcode, x, y, w, h = one_label_str.split(' ')

        labels.append([charcode, int(x), int(y), int(w), int(h)])

    return labels



def visualize_training_data(image_path, labels):

    fs = 8

    # Read image

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    for label in convert_labels_set(labels):

        _, x, y, w, h = label

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    return img
n_sheets = 2



for _ in range(n_sheets):

    img_filename, labels = df_train.values[np.random.randint(len(df_train))]

    viz_img = visualize_training_data('../input/kuzushiji-recognition/train_images/{}.jpg'.format(img_filename), labels)

    

    plt.figure(figsize=(15, 15))

    plt.title(img_filename)

    plt.imshow(viz_img)

    plt.show()
n_labels = 0

chars_counts = {}



for labels_set in df_train.values[:, 1]:

    if type(labels_set) is not str:

        continue



    labels = convert_labels_set(labels_set)

    n_labels += len(labels)

    for label in labels:

        try:

            chars_counts[label[0]] += 1

        except KeyError:

            chars_counts.update({label[0]: 1})
chars_counts_list = [chars_counts[k] for k in chars_counts]

n_classes = len([k for k in chars_counts])
print('Number of labels:                  {}'.format(n_labels))

print('Number of classes:                 {}'.format(n_classes))

print('Min max number of items per class: {} {}'.format(np.min(chars_counts_list), np.max(chars_counts_list)))

print('Median number of items per class:  {}'.format(np.median(chars_counts_list)))

print('Mean number of items per class:    {}'.format(np.mean(chars_counts_list)))
def get_char_images_from_sheet(src_image_path, labels_str, blur_kernel_size=3, img_size=64):

    src_img = cv2.imread(src_image_path, cv2.IMREAD_COLOR)



    char_imgs = []

    for label in convert_labels_set(labels_str):

        char_img = np.zeros((img_size, img_size), dtype=np.uint8)

        _, x, y, w, h = label



        label_img = src_img[y:y + h, x:x + w, :]

        label_img = cv2.GaussianBlur(label_img, 

                                     (blur_kernel_size, blur_kernel_size), 

                                     cv2.BORDER_DEFAULT)

        label_img = cv2.cvtColor(label_img, cv2.COLOR_RGB2GRAY)

        _, label_img = cv2.threshold(label_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        label_img = 255 - label_img

        

        if w > h:

            label_img = cv2.resize(label_img, (img_size, int(img_size * h / w)))

            dy = int((img_size - int(img_size * h / w)) / 2)

            char_img[dy:dy + int(img_size * h / w), :] += label_img

        

        else:

            label_img = cv2.resize(label_img, (int(img_size * w / h), img_size))            

            dx = int((img_size - int(img_size * w / h)) / 2)

            char_img[:, dx:dx + int(img_size * w / h)] += label_img

        

        char_imgs.append(char_img)

    return char_imgs
img_filename, labels = df_train.values[np.random.randint(len(df_train))]



char_imgs = get_char_images_from_sheet('../input/kuzushiji-recognition/train_images/{}.jpg'.format(img_filename), labels)



for i in np.random.choice(len(char_imgs), 10):

    plt.figure(figsize=(2, 2))

    plt.imshow(char_imgs[i], cmap='Greys')

    plt.show()