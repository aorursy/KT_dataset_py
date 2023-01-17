# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import cv2
import random
import shutil
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fake_input = '/kaggle/input/1-million-fake-faces/1m_faces_00_01_02_03/1m_faces_00_01_02_03/1m_faces_02'
real_input = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'
fake_save_dir = '/kaggle/tmp/fake'
real_save_dir = '/kaggle/tmp/real'
if not os.path.exists(fake_save_dir):
    os.makedirs(fake_save_dir)
if not os.path.exists(real_save_dir):
    os.makedirs(real_save_dir)
real_images = random.sample(os.listdir(real_input),500)
real_paths = []
for image in real_images:
    real_path = os.path.join(real_input,image)
    real_paths.append(real_path)
fake_images = random.sample(os.listdir(fake_input),500)
fake_paths = []
for image in fake_images:
    fake_path = os.path.join(fake_input,image)
    fake_paths.append(fake_path)
def move_resize_images(save_dir,path_list):
    i = 0
    for path, filename in tqdm(path_list):
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        filename = str(i) + '.jpg'
        cv2.imwrite(os.path.join(save_dir, filename), img)
        i += 1
for path in real_paths:
    filename = os.path.basename(path)
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    cv2.imwrite(os.path.join(real_save_dir,filename),img)
for path in fake_paths:
    filename = os.path.basename(path)
    img = cv2.imread(path)
    img = cv2.resize(img,(224,224))
    cv2.imwrite(os.path.join(fake_save_dir,filename),img)
ls /kaggle/tmp
alexnet_weights_path = '/kaggle/input/assignment3alexnet/weights.28-0.23.hdf5'
alexnet = tf.keras.models.load_model(weights_path)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '/kaggle/tmp',
    target_size=(224,224),
    batch_size=50,
    class_mode='binary',
    shuffle=False
)
alexnet.evaluate(test_generator)
vgg16_weghts_path = '/kaggle/input/deepfake-vgg16-weights-017-loss/weights.08-0.17.hdf5'
vgg16 = tf.keras.models.load_model(vgg16_weghts_path)
vgg16.evaluate(test_generator)
