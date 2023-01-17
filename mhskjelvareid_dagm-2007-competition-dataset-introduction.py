# Imports

import numpy as np # linear algebra

import os

import glob

import matplotlib.pyplot as plt
# Paths

base_dir = "../input/dagm_kaggleupload/DAGM_KaggleUpload/"

classes = ["Class" + str(i+1) for i in range(10)]

dev_classes = classes[:6]

comp_classes = classes[6:]

print(dev_classes)

print(comp_classes)
# Create lists with paths for training and test images (with labeled defect images)

trn_imdir  = []  

trn_ims    = []

trn_labdir = []

trn_labims = []



tst_imdir  = []

tst_ims    = []

tst_labdir = []

tst_labims = []



for cls in classes:

    # Get file names for all training images and label images

    trn_imdir.append(os.path.join(base_dir, cls, "Train"))

    trn_ims.append(sorted(glob.glob(trn_imdir[-1] + "/*.PNG")))

    

    trn_labdir.append(os.path.join(base_dir, cls, "Train", "Label"))

    trn_labims.append(sorted(glob.glob(trn_labdir[-1] + "/*.PNG")))



    tst_imdir.append(os.path.join(base_dir, cls, "Test"))

    tst_ims.append(sorted(glob.glob(tst_imdir[-1] + "/*.PNG")))

    

    tst_labdir.append(os.path.join(base_dir, cls, "Test", "Label"))

    tst_labims.append(sorted(glob.glob(tst_labdir[-1] + "/*.PNG")))
print(trn_labdir[5])
# Show example images with corresponding label image

for i in range(len(classes)):

    example_defect_image = os.path.basename(trn_labims[i][0])

    im = plt.imread(trn_imdir[i] + "/" + example_defect_image[0:4] + ".PNG")

    labim = plt.imread(trn_labdir[i] + "/" + example_defect_image)

    plt.figure(i,[12,9])

    plt.subplot(121)

    plt.imshow(im,"gray")

    plt.subplot(122)

    plt.imshow(labim,"gray")
import keras

help(keras.applications.VGG16())