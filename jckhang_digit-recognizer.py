import numpy as np 
import pandas as pd
import numpy
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline
# settings
LEARNING_RATE = 1e-4
COMPONENT_NUM = 35
# set to 20000 on local environment to get 0.99 accuracy
TRAINING_ITERATIONS = 2500        
    
DROPOUT = 0.5
BATCH_SIZE = 50

# set to 0 to train on all available data
VALIDATION_SIZE = 2000

# image number to output
IMAGE_TO_DISPLAY = 10
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
images = train.iloc[:,1:].values
images = images.astype(np.float)
images = np.multiply(images, 1.0 / 255.0)

print('images({0[0]},{0[1]})'.format(images.shape))
image_size = images.shape[1]
print ('image_size => {0}'.format(image_size))

# in this case all images are square
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
# display image
def display(img):
    
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)
    
    plt.axis('off')
    plt.imshow(one_image, cmap=cm.binary)

# output image     
display(images[IMAGE_TO_DISPLAY])
def display_loop(img,temp):
    
    # (784) => (28,28)
    one_image = img.reshape(image_width,image_height)

    plt.subplot(2,5,temp)
    
    plt.imshow(one_image, cmap=cm.binary)
    plt.axis('off')

for i in range(10):
    index = train[train['label']==i].index
    display_loop(images[index].mean(0), i+1)
# 1. average number of non-zero pixels
for i in range(10):
    index = train[train['label']==i].index
    images_sub = images[index]
    ave = 0
    for image in images_sub:
        ave += (len(image)-list(image).count(0))/len(index)
    print("For {}, the average number of non-zero pixels is {}.".format(i, ave))
# 2. average number of zero pixels
for i in range(10):
    index = train[train['label']==i].index
    images_sub = images[index]
    ave = 0
    for image in images_sub:
        ave += list(image).count(0)/len(index)
    print("For {}, the average number of zero pixels is {}.".format(i, ave))
# 3. ratio of non-zero to zero
for i in range(10):
    index = train[train['label']==i].index

    images_sub = images[index]
    ave = 0
    for image in images_sub:
        ave += ((len(image)-list(image).count(0))/list(image).count(0))/len(index)
    print("For {}, the ratio of non-zero to zero is {}.".format(i, ave))
# Add Features: non-zero, zero, ratio
image_withFeature = []
for image in images:
    newLine = list(image)
    newLine.append(len(image)-list(image).count(0)) # Add non-zero
    newLine.append(list(image).count(0)) # Add zero
    newLine.append((len(image)-list(image).count(0))/list(image).count(0)) # Add ratio
    image_withFeature.append(newLine)
# PCA
pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(train)
train_pca = pca.transform(train)
image_withFeature
