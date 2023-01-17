from __future__ import division, print_function

# Rather than importing everything manually, we'll make things easy

#   and load them all in utils.py, and just import them from there.

%matplotlib inline

from importlib import reload  # Python 3

import utils; reload(utils)

from utils import *

# utils are from fast.ai course: https://github.com/fastai/courses/tree/master/deeplearning1/nbs
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "/home/fabsta/projects/datascience/competitions/kaggle_plant_seedlings_classification/data/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path = "../kaggle_plant_seedlings_classification/data/"

model_path = path + 'models/'

train_path = path + 'train/'

val_path = path + 'val/'

test_path = path + 'test/'

results_path=path + 'results/'



if not os.path.exists(model_path): os.mkdir(model_path)

if not os.path.exists(val_path): os.mkdir(val_path)

if not os.path.exists(results_path): os.mkdir(results_path)

    
val_split = 0.2



for dir in os.listdir(train_path):

    src_dir,val_dir = (os.path.join(train_path, dir),os.path.join(val_path, dir))

    no_files_to_move = int(len(os.listdir(src_dir)) * val_split)

    counter = 0

    for file in os.listdir(src_dir):

        if counter > no_files_to_move:

            break

        if os.path.isfile(os.path.join(src_dir,file)):

            src_file,dest_file = (os.path.join(src_dir, file),os.path.join(val_dir, file))

            #print(src_file,dest_file)

            os.rename(src_file,dest_file)

            counter = counter + 1

batch_size=16
#from keras import backend as K

#K.set_image_dim_ordering('th')
# Import our class, and instantiate

import vgg16; reload(vgg16)

from vgg16 import Vgg16
no_of_epochs = 5

#vgg = Vgg16()

vgg = Vgg16(weights='imagenet', include_top=False)

# Grab a few images at a time for training and validation.

# NB: They must be in subdirectories named based on their category

batches = vgg.get_batches(path+'train', batch_size=batch_size)

val_batches = vgg.get_batches(path+'val', batch_size=batch_size*2)

vgg.finetune(batches)

#vgg.fit(batches, val_batches, batch_size, nb_epoch=3)



latest_weights_filename = None

for epoch in range(no_of_epochs):

    print("Running epoch: %d" % epoch)

    vgg.fit(batches, val_batches, batch_size, nb_epoch=1)

    latest_weights_filename = 'ft%d.h5' % epoch

    vgg.model.save_weights(results_path+latest_weights_filename)

print("Completed %s fit operations" % no_of_epochs)
batches, preds = vgg.test(test_path, batch_size = batch_size*2)
print(preds[:5])

filenames = batches.filenames

print(filenames[:5])
#Save our test results arrays so we can use them again later

save_array(results_path + 'test_preds.dat', preds)

save_array(results_path + 'filenames.dat', filenames)
vgg.model.load_weights(results_path+latest_weights_filename)
batches, probs = vgg.test(val_path, batch_size = batch_size)
filenames = batches.filenames

expected_labels = batches.classes #0 or 1



#Round our predictions to 0/1 to generate labels

our_predictions = probs[:,0]

our_labels = np.round(1-our_predictions)
from keras.preprocessing import image



#Helper function to plot images by index in the validation set 

#Plots is a helper function in utils.py

def plots_idx(idx, titles=None):

    plots([image.load_img(val_path + filenames[i]) for i in idx], titles=titles)

    

#Number of images to view for each visualization task

n_view = 4
#1. A few correct labels at random

correct = np.where(our_labels==expected_labels)[0]

print("Found %d correct labels" % len(correct))

idx = permutation(correct)[:n_view]

plots_idx(idx, our_predictions[idx])
#2. A few incorrect labels at random

incorrect = np.where(our_labels!=expected_labels)[0]

print("Found %d incorrect labels" % len(incorrect))

idx = permutation(incorrect)[:n_view]

plots_idx(idx, our_predictions[idx])
#5. The most uncertain labels (ie those with probability closest to 0.5).

most_uncertain = np.argsort(np.abs(our_predictions-0.5))

plots_idx(most_uncertain[:n_view], our_predictions[most_uncertain])
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(expected_labels, our_labels)

plot_confusion_matrix(cm, batches.class_indices)
#Load our test predictions from file

preds = load_array(results_path + 'test_preds.dat')

filenames = load_array(results_path + 'filenames.dat')
#filenames = batches.filenames

ids = np.array([f[8:f.find('.')] for f in filenames])




label_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']





#label_names = ['Black-grass','Cleavers','Common wheat','Loose Silky-bent',

#     'Scentless Mayweed','Small-flowered Cranesbill','Charlock','Common Chickweed',

#     'Fat Hen','Maize','Shepherds Purse','Sugar beet']

label_names[np.argmax(preds[:1])]

type(preds)

pred_labels = np.array(label_names[np.argmax(preds[x])] for x,item in enumerate(preds))

labels = []

for x,item in enumerate(preds):

    #print(x)

    labels = np.append(labels, label_names[np.argmax(preds[x])])



#subm = np.stack([ids,isdog], axis=1)

#subm[:5]
print(labels.shape)

print(ids.shape)
subm = np.stack([ids,labels], axis=1)

subm[:5]
df = pd.DataFrame({'file' : ids,'species' : labels})

df['file'] = df.file + '.png'

#df['col'] = 'str' + df['col'].astype(str)

df = df.sort_values('file')
df.to_csv('submission1.csv', encoding='utf-8', index=False)
from IPython.display import FileLink

#%cd $LESSON_HOME_DIR

FileLink('submission1.csv')