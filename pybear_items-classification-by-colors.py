import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import keras

import os

import warnings

import cv2



warnings.filterwarnings('ignore')
main_file = "../input/6000-store-items-images-classified-by-color"



available_files = os.listdir(main_file)

print("Available files are: \n\t{}".format(available_files))
training = os.path.join(main_file, 'train')

testing = os.path.join(main_file, 'test')
training_images = {category: [] for category in os.listdir(os.path.join(training))}

testing_images = []



labels = {ix: category for ix, category in enumerate(list(training_images.keys()))}
n_H = 160

n_W = 160

n_C = 3



for file in ['training', 'testing']:

    for category in range(12):

        if file == 'training':

            for image in os.listdir(os.path.join(training, labels[category])):

                img = plt.imread(os.path.join(training, labels[category], image))

                training_images[labels[category]].append((cv2.resize(img, (n_H, n_W)), category))

        else:

            for image in os.listdir(os.path.join(testing)):

                img = plt.imread(os.path.join(testing, image))

                testing_images.append(cv2.resize(img, (n_H, n_W)))
print("Training set ...")



for category in range(12):

    print("There are {} image in the category of {}".format(len(training_images[labels[category]]), 

                                                            labels[category]))
Index = 3

cate = np.random.choice([int(i) for i in range(12)])

image = training_images[labels[cate]][Index][0]

target = training_images[labels[cate]][Index][1]



plt.imshow(image)

plt.title("Label: {}".format(labels[target]))

plt.xticks([])

plt.yticks([])

plt.show()

def get_inputs(training_images, train_size = None, dev_size = None) :

    m_examples = 0

    for category in list(training_images.keys()) : 

        m_examples += len(training_images[category])

    

    m_train = np.floor(np.multiply(train_size, m_examples))

    m_dev = np.subtract(m_examples, m_train)

    

    x_train = np.zeros([int(m_train), n_H, n_W, n_C])

    y_train = np.zeros([1, int(m_train)])

    

    x_dev = np.zeros([int(m_dev), n_H, n_W, n_C])

    y_dev = np.zeros([1, int(m_dev)])

    

    all_images = []

    for category in range(12) : 

        for item in training_images[labels[category]] : 

            all_images.append(item)

    np.random.shuffle(all_images)

    

    for train_image in range(int(m_train)) : 

        x_train[train_image] = all_images[train_image][0]

        y_train[:, train_image] = all_images[train_image][1] 

        

    #for dev_image in range(int(m_dev)) :

    #    x_dev[dev_image] = all_images[dev_image][0]

        #y_dev[dev_image] = all_images[dev_image][1]

    

    return x_train, y_train
x_train, y_train = get_inputs(training_images, train_size = 0.8, dev_size = 0.2)