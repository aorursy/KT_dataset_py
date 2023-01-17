from random import shuffle
import glob
import pandas as pd
import cv2


df = pd.read_csv("train.csv")
shuffle_data = False  # shuffle the addresses before saving
hdf5_path = 'dataset.hdf5'  # address to where you want to save the hdf5 file
# read addresses and labels from the 'train' folder
addrs = df.Image.apply(lambda x: "train/" + x).values
labels = df.Id.values  # 0 = Cat, 1 = Dog
img_names = df.Image.values
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels,img_names))
    shuffle(c)
    train_addrs, train_labels, train_img_name = zip(*c)
else:
    train_addrs = list(addrs)
    train_labels = list(labels)
    train_img_name = list(img_names)
import numpy as np
import tables

data_order = 'tf'  # 'th' for Theano, 'tf' for Tensorflow
img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved
# check the order of data and chose proper data shape to save images
if data_order == 'th':
    data_shape = (0, 3, 224, 224)
elif data_order == 'tf':
    data_shape = (0, 128,256, 3)
# open a hdf5 file and create earrays
hdf5_file = tables.open_file(hdf5_path, mode='w')
train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)
# create the label arrays and copy the labels data in them
hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
hdf5_file.create_array(hdf5_file.root, 'train_img_name', train_img_name)

# a numpy array to save the mean of the images
mean = np.zeros(data_shape[1:], np.float32)
# loop over train addresses
for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if i % 1000 == 0 and i > 1:
        print('Train data: ' + str(i) + "/" + str(len(train_addrs)))
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (256, 128))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    if data_order == 'th':
        img = np.rollaxis(img, 2)
    # save the image and calculate the mean so far
    train_storage.append(img[None])
    mean += img / float(len(train_labels))
# save the mean and close the hdf5 file
mean_storage.append(mean[None])
hdf5_file.close()
hdf5_path = 'dataset.hdf5'
subtract_mean = False
# open the hdf5 file
hdf5_file = tables.open_file(hdf5_path, mode='r')
# subtract the training mean
if subtract_mean:
    mm = hdf5_file.root.train_mean[0]
    mm = mm[np.newaxis, ...]
# Total number of samples
data_num = hdf5_file.root.train_img.shape[0]
hdf5_file.root.train_img[:10].shape
hdf5_file.root.train_img_name[:10]
hdf5_file.root.train_labels[:10]
