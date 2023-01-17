import os

import glob

from time import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import imageio as io

import cv2 as cv

import matplotlib.pyplot as plt

%matplotlib inline

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



print("Setup complete!")



# Input data files are available in the "../input/" directory.
def convert_seconds_to_time(seconds):

    """

    (float -> str)

    

    Converts seconds (float) in days, hours, minutes and seconds and returns a string with the result.  

    """

    if seconds < float(86400) and seconds >= float(3600):

        h, sec = divmod(int(round(seconds)), 3600)

        m, sec = divmod(int(sec), 60)

        return f'{int(h)} hours, {int(m)} minutes and {round(sec)} seconds'

    

    elif seconds < float(86400) and seconds < float(3600):

        if seconds >= float(60):

            m, sec = divmod(int(round(seconds)), 60)

            return f'{int(m)} minutes and {round(sec)} seconds'

        else:

            return f'{round(seconds)} seconds'

    else:

        d, sec = divmod(int(round(seconds)), 86400)

        return f'{int(d)} days, {convert_seconds(float(sec))}'



def diab_retin(prediction):

    """

    (int -> str)

    

    Returns a string with information of the type of diabetic retinopathy, if present, 

    according to an integer which is the prediction given by the model.

    """

    if prediction == 0:

        return "No diabetic retinopathy"

    elif prediction == 1:

        return "Mild non-proliferative diabetic retinopathy"

    elif prediction == 2:

        return "Moderate non-proliferative diabetic retinopathy"

    elif prediction == 3:

        return "Severe non-proliferative diabetic retinopathy"

    elif prediction == 4:

        return "Proliferative diabetic retinopathy"

    else:

        raise ValueError("The argument should be an integer from 0 to 4, both included.")
print("Number of images in the training set:", len(os.listdir("../input/train_images")))

print("Number of images in the test set:", len(os.listdir("../input/test_images")))
url = r"../input/train.csv"

train = pd.read_csv(url)



print("One-hot encoding of the provided labels...", end = " ")

y_train_noncat = train["diagnosis"].values

y_train_cat = to_categorical(y_train_noncat, 5)

print("Done!")
train_id_codes = train["id_code"].tolist()

list_train_img = []

a = 0



timea = time() 

print("Converting and adding training images to a 4-d tensor (batch_size, height, width, channels)...")

for im in train_id_codes:

    uri = glob.glob("../input/train_images/" + im + ".*")

    image = io.imread(uri[0])

    image = cv.resize(image, (256, 256), interpolation = cv.INTER_AREA) / 255 #Normalising...

    list_train_img.append(image)

    a += 1

    if a % 500 == 0:

        print(f"\t{a} images from the training set converted and added to the tensor")

        

timeb = time()

total_time = timeb- timea

print(f"\nIt took {convert_seconds_to_time(total_time)} to complete the process")

print("All images from the training set converted and added!")



train_im = np.asarray(list_train_img)

print("The shape of the input training set before splitting is", train_im.shape)

print("The shape of the array containing the labels of the training set before splitting is", y_train_cat.shape)

del list_train_img

#rand_samples = np.random.randint(0, 3663, size = 3)



#fig, arr = plt.subplots(1,3, figsize = (20, 10), sharey = "all")

#plt.suptitle("3 random examples from the provided training dataset", fontsize = 20)



#arr[0].imshow(train_im[rand_samples[0]])

#arr[0].set_title(diab_retin(y_train_noncat[rand_samples[0]]))

#arr[0].axis("off")

#arr[1].imshow(train_im[rand_samples[1]])

#arr[1].set_title(diab_retin(y_train_noncat[rand_samples[1]]))

#arr[1].axis("off")

#arr[2].imshow(train_im[rand_samples[2]])

#arr[2].set_title(diab_retin(y_train_noncat[rand_samples[2]]))

#arr[2].axis("off")



#plt.show()
#NOW THE SAME FOR THE TEST SET OF IMAGES:

url = r"../input/test.csv"

test = pd.read_csv(url)

test_id_codes = test["id_code"].tolist()



list_test_img = []

b = 0



timec = time()

print("Converting and adding test images to a 4-D tensor (batch_size, height, width, channels)...")

for im_test in test_id_codes:

    uri = glob.glob("../input/test_images/" + im_test + ".*")

    image = io.imread(uri[0])

    image = cv.resize(image, (256, 256), interpolation = cv.INTER_AREA) / 255 #Normalising...

    list_test_img.append(image)

    b += 1

    if b % 500 == 0:

        print(f"\t{b} images from the test set converted and added to the tensor")      



timed = time()

total_time = timed - timec

print(f"It took {convert_seconds_to_time(total_time)} to complete the process")        

print("All images from the test set converted and added!\n")



test_im = np.asarray(list_test_img)

print("The shape of the test set is", test_im.shape)

del list_test_img
print("Splitting input tensor and labels into random train and cross-validation subsets...", end = " ")

X_train, X_val, Y_train, Y_val = train_test_split(train_im, y_train_cat, test_size = 0.2)

print("Done!\n")

print("Shape of the training set inputs (X_train):", X_train.shape)

print("Shape of the cross-validation set (X_val):", X_val.shape)

print("Shape of the training subset labels after one-hot encoding (Y_train):", Y_train.shape)

print("Shape of the cross-validation subset labels after one-hot encoding (Y_val):", Y_val.shape)
np.savez_compressed(file = "train_set", X_train = X_train, Y_train = Y_train)

np.savez_compressed(file = "val_set", X_val = X_val, Y_val = Y_val)

np.savez_compressed(file = "input_dataset", train_im = train_im)

np.savez_compressed(file = "dataset_labels", dataset_labels = y_train_cat)

np.savez_compressed(file = "test_set", test_im = test_im)



print("Name of the file containing the full provided set of images: input_dataset.npz")

print("Name of the file containing all the provided labels: dataset_labels.npz")

print("Name of the file containing the training subset (inputs and labels): train_set.npz")

print("Name of the file containing the cross-validation subset (inputs and labels): val_set.npz")

print("Name of the file containing the test set of images: test_set.npz")