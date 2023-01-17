import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt



# These paths are unique to Kaggle, obviously. Use your local path or colab path, depending on which you're using.

train_x = np.load('/kaggle/input/f2019-aihw7/mnist-train-images.npy')

train_y = np.load('/kaggle/input/f2019-aihw7/mnist-train-labels.npy')

val_x = np.load('/kaggle/input/f2019-aihw7/mnist-val-images.npy')

val_y = np.load('/kaggle/input/f2019-aihw7/mnist-val-labels.npy')



# Verify that their shapes are what we expect

print("train_x shape:", train_x.shape)

print("train_y shape:", train_y.shape)

print("val_x shape:", val_x.shape)

print("val_y shape:", val_y.shape)
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)



# show each image, and make each title the label

# these are grayscale images so use appropriate heatmap

ax1.imshow(train_x[4701], cmap=plt.get_cmap('gray'))

ax1.set_title(str(train_y[4701]))

ax2.imshow(train_x[4702], cmap=plt.get_cmap('gray'))

ax2.set_title(str(train_y[4702]))

ax3.imshow(train_x[4703], cmap=plt.get_cmap('gray'))

ax3.set_title(str(train_y[4703]))



fig.show()
# print data type

print("Data type:", train_x.dtype)

# just to make sure, print the min/max too

print("Data min:", np.amin(train_x[4701]))

print("Data max:", np.amax(train_x[4701]))
print("Data type:", train_y.dtype)
fig, ax = plt.subplots()

ax.hist(train_y, bins=range(11))

ax.set_xticks(range(10))

ax.set_title("MNIST Training Set Class Distribution")



fig.show()