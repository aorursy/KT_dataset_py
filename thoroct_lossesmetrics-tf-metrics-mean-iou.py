# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import itertools

import tensorflow as tf

import matplotlib.pyplot as plt

# tf.enable_eager_execution()

tf.__version__
def iou(A1, A2):

    return (np.sum(np.logical_and(A1, A2)))/np.sum(np.logical_or(A1, A2))

def ds(A1, A2):

    return (2*np.sum(A1*A2))/(np.sum(A1) + np.sum(A2))
# To define circular lable we need to first define x- y- grids.

my, mx = np.mgrid[0:256, 0:256]

# Then we define center_y, center_x, radius for both true and predicted labels of category A. 

cay1, cax1, ra1, cay2, cax2, ra2  = 156, 200, 120, 156, 200, 124  # constant area difference A

A1 = np.less(np.power(my-cay1, 2) + np.power(mx-cax1, 2), ra1**2) 

A2 = np.less(np.power(my-cay2, 2) + np.power(mx-cax2, 2), ra2**2) 



# Then we define center_y, center_x, radius for both true and predicted labels of category B.

cby1, cbx1, rb1, cby2, cbx2, rb2  = 56, 28, 28, 56, 28, 41.95 # constant area difference B

B1 = np.less(np.power(my-cby1, 2) + np.power(mx-cbx1, 2), rb1**2) 

B2 = np.less(np.power(my-cby2, 2) + np.power(mx-cbx2, 2), rb2**2) 



# Calculations

IoU_A = iou(A1, A2)

IoU_B = iou(B1, B2)

DS_A = ds(A1, A2)

DS_B = ds(B1, B2)



# Print values. From the values below, we observe that when the difference in areas, numbers are rather low in small regions. 

print(f"Overlapping indices are: \nIoU_A {IoU_A}, IoU_B {IoU_B}, DS_A {DS_A}, DS_B {DS_B} \nmean IoU over 1 and 2 is {(IoU_A+IoU_B)/2}, \nmean DS over 1 and 2 is {(DS_A+DS_B)/2}")
my, mx = np.mgrid[0:256, 0:256]

cay1, cax1, ra1, cay2, cax2, ra2  = 156, 200, 120, 157, 189, 118

A1 = np.less(np.power(my-cay1, 2) + np.power(mx-cax1, 2), ra1**2) 

A2 = np.less(np.power(my-cay2, 2) + np.power(mx-cax2, 2), ra2**2) 



cby1, cbx1, rb1, cby2, cbx2, rb2  = 56, 28, 28, 58, 30, 29 

B1 = np.less(np.power(my-cby1, 2) + np.power(mx-cbx1, 2), rb1**2) 

B2 = np.less(np.power(my-cby2, 2) + np.power(mx-cbx2, 2), rb2**2) 



C1 = np.logical_not(np.logical_or(A1, B1))

C2 = np.logical_not(np.logical_or(A2, B2))



IoU_A = iou(A1, A2)

IoU_B = iou(B1, B2)

IoU_C = iou(C1, C2)

DS_A = ds(A1, A2)

DS_B = ds(B1, B2)

DS_C = ds(C1, C2)



print(f"Overlapping indices are: \nIoU_A {IoU_A}\nIoU_B {IoU_B}\nIoU_C {IoU_C}")

print(f"DS_A {DS_A}\nDS_B {DS_B}\nDS_C {DS_C}, \nmean IoU over 1 and 2 is {(IoU_A+IoU_B+IoU_C)/3}, \nmean DS over 1 and 2 is {(DS_A+DS_B+DS_C)/3}")



plt.rcParams['figure.figsize'] = [16, 6]

fig, axes = plt.subplots(2, 6)

axes[0, 0].set_frame_on(False)

axes[0, 1].set_frame_on(False)

axes[0, -2].set_frame_on(False)

axes[0, -1].set_frame_on(False)

axes[0, 2].imshow(mx)

axes[0, 2].set_title('x grid')

axes[0, 3].imshow(my)

axes[0, 3].set_title('y grid')

axes[1, 0].imshow(A1)

axes[1, 0].set_title('region A in true label')

axes[1, 1].imshow(A2)

axes[1, 1].set_title('region A in pred label')

axes[1, 2].imshow(B1)

axes[1, 2].set_title('region B in true label')

axes[1, 3].imshow(B2)

axes[1, 3].set_title('region B in pred label')

axes[1, 4].imshow(C1)

axes[1, 4].set_title('region C in true label')

axes[1, 5].imshow(C2)

axes[1, 5].set_title('region C in pred label')

_ = [axes[i, j].axis('off') for i, j in itertools.product(range(2), range(6))]

plt.subplots_adjust(left = 0.01,  # the left side of the subplots of the figure

                    right = 0.99,   # the right side of the subplots of the figure

                    bottom = 0.01,  # the bottom of the subplots of the figure

                    top = 0.99,     # the top of the subplots of the figure

                    wspace = 0.02,  # the amount of width reserved for space between subplots,

                                    # expressed as a fraction of the average axis width

                    hspace = 0.05)  # the amount of height reserved for space between subplots,

                                    # expressed as a fraction of the average axis height

plt.show()
# Logical labels of Region C, A and B are stacked

label = np.concatenate((np.zeros(C1[None, :,:, None].shape), A1[None, :,:, None], B1[None, :,:, None]), axis=3).astype(float)

prediction = np.concatenate((np.zeros(C2[None, :,:, None].shape), A2[None, :,:, None],  B2[None, :,:, None]), axis=3).astype(float)



# Logical labels were converted into integer label (See the output figure)

trueL = np.argmax(label, axis=-1)

predL = np.argmax(prediction, axis=-1)

plt.rcParams['figure.figsize'] = [8, 4]

plt.subplot(1, 2, 1)

plt.imshow(trueL[0])

plt.axis('off')

plt.title('true label', fontdict={'fontsize': 20})

plt.subplot(1, 2, 2)

plt.imshow(predL[0])

plt.axis('off')

plt.title('pred label', fontdict={'fontsize': 20})

plt.subplots_adjust(wspace = 0.02,  hspace = 0.02) 



# run tf.metrics.mean_iou

trueT = tf.constant(trueL)

predT = tf.constant(predL)

mean_iou, conf_mat = tf.metrics.mean_iou(trueT, predT, num_classes=3)

with tf.Session() as sess:    

    sess.run(tf.local_variables_initializer())

    sess.run(tf.global_variables_initializer())

    sess.run([conf_mat])

    res = sess.run(mean_iou)

print(f"the mean IoU given by tf.metrics.mean_iou is: {res}")
# Logical labels of A and B are stacked

label = np.concatenate((A1[None, :,:, None], B1[None, :,:, None]), axis=3).astype(float)

prediction = np.concatenate((A2[None, :,:, None], B2[None, :,:, None]), axis=3).astype(float)

print(f"region A and B were taken as one label and the IoU is: {iou(label, prediction)}")

print(f"region of zero-label yields an IoU of: {iou(np.logical_not(label), np.logical_not(prediction))}")

print(f"mean IoU of two calculations is: {0.5*(iou(label, prediction) + iou(np.logical_not(label), np.logical_not(prediction)))}")

# run tf.metrics.mean_iou

trueT = tf.constant(label.reshape(-1))

predT = tf.constant(prediction.reshape(-1))

mean_iou, conf_mat = tf.metrics.mean_iou(trueT, predT, num_classes=2)

with tf.Session() as sess:    

    sess.run(tf.local_variables_initializer())

    sess.run(tf.global_variables_initializer())

    sess.run([conf_mat])

    res = sess.run(mean_iou)

print(f"the mean IoU given by tf.metrics.mean_iou is: {res}")
my, mx = np.mgrid[0:256, 0:256]

cay1, cax1, ra1, cay2, cax2, ra2  = 156, 200, 91, 157, 189, 80

A1 = np.less(np.power(my-cay1, 2) + np.power(mx-cax1, 2), ra1**2) 

A2 = np.less(np.power(my-cay2, 2) + np.power(mx-cax2, 2), ra2**2) 



cby1, cbx1, rb1, cby2, cbx2, rb2  = 56, 28, 100, 58, 30, 100 

B1 = np.less(np.power(my-cby1, 2) + np.power(mx-cbx1, 2), rb1**2) 

B2 = np.less(np.power(my-cby2, 2) + np.power(mx-cbx2, 2), rb2**2) 



C1 = np.logical_not(np.logical_or(A1, B1))

C2 = np.logical_not(np.logical_or(A2, B2))



IoU_A = iou(A1, A2)

IoU_B = iou(B1, B2)

IoU_C = iou(C1, C2)



print(f"Overlapping indices are: \nIoU_A {IoU_A}\nIoU_B {IoU_B}\nIoU_C {IoU_C}")

print(f"mean IoU over 1 and 2 (region  A, B, C) is {(IoU_A+IoU_B+IoU_C)/3}")

print(f"if we only consider region A and B the mean IoU should be: {0.5*(IoU_A+IoU_B)}")





plt.rcParams['figure.figsize'] = [16, 6]

fig, axes = plt.subplots(2, 6)

axes[0, 0].set_frame_on(False)

axes[0, 1].set_frame_on(False)

axes[0, -2].set_frame_on(False)

axes[0, -1].set_frame_on(False)

axes[0, 2].imshow(mx)

axes[0, 2].set_title('x grid')

axes[0, 3].imshow(my)

axes[0, 3].set_title('y grid')

axes[1, 0].imshow(A1)

axes[1, 0].set_title('region A in true label')

axes[1, 1].imshow(A2)

axes[1, 1].set_title('region A in pred label')

axes[1, 2].imshow(B1)

axes[1, 2].set_title('region B in true label')

axes[1, 3].imshow(B2)

axes[1, 3].set_title('region B in pred label')

axes[1, 4].imshow(C1)

axes[1, 4].set_title('region C in true label')

axes[1, 5].imshow(C2)

axes[1, 5].set_title('region C in pred label')

_ = [axes[i, j].axis('off') for i, j in itertools.product(range(2), range(6))]

plt.subplots_adjust(left = 0.01,  # the left side of the subplots of the figure

                    right = 0.99,   # the right side of the subplots of the figure

                    bottom = 0.01,  # the bottom of the subplots of the figure

                    top = 0.99,     # the top of the subplots of the figure

                    wspace = 0.02,  # the amount of width reserved for space between subplots,

                                    # expressed as a fraction of the average axis width

                    hspace = 0.05)  # the amount of height reserved for space between subplots,

                                    # expressed as a fraction of the average axis height

plt.show()

# Logical labels of Region C, A and B are stacked

label = np.concatenate((np.zeros(C1[None, :,:, None].shape), A1[None, :,:, None], B1[None, :,:, None]), axis=3).astype(float)

prediction = np.concatenate((np.zeros(C2[None, :,:, None].shape), A2[None, :,:, None],  B2[None, :,:, None]), axis=3).astype(float)

# weights = np.concatenate((np.zeros(C1[None, :,:, None].shape), np.ones(A1[None, :,:, None].shape), np.ones(B1[None, :,:, None].shape)), axis=3).astype(float)

# Logical labels were converted into integer label (See the output figure)

trueL = np.argmax(label, axis=-1)

predL = np.argmax(prediction, axis=-1)

weights = np.logical_or(np.greater(trueL, 0),np.greater(predL, 0)).astype(float)

# run tf.metrics.mean_iou

trueT = tf.constant(trueL)

predT = tf.constant(predL)

mean_iou, conf_mat = tf.metrics.mean_iou(trueT, predT, num_classes=3, weights=tf.math.logical_or(tf.math.greater(trueT, 0), tf.math.greater(predT, 0)))

with tf.Session() as sess:    

    sess.run(tf.local_variables_initializer())

    sess.run(tf.global_variables_initializer())

    sess.run([conf_mat])

    res = sess.run(mean_iou)

print(f"the mean IoU given by tf.metrics.mean_iou is: {res*3/2}")
