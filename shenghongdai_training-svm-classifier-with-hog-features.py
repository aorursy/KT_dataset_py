# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 

import json

from matplotlib import pyplot as plt

from skimage import color

from skimage.feature import hog

from sklearn import svm

from sklearn.metrics import classification_report,accuracy_score



import matplotlib.pyplot as plt # plt 用于显示图片

import matplotlib.image as mpimg # mpimg 用于读取图片



import keras

from keras.utils import to_categorical



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ROOT_DIR = '../input/state-farm-distracted-driver-detection/'

TRAIN_DIR = ROOT_DIR + 'imgs/train/'

TEST_DIR = ROOT_DIR + 'imgs/test/'

driver_imgs_list = pd.read_csv(ROOT_DIR + "driver_imgs_list.csv")

sample_submission = pd.read_csv(ROOT_DIR + "sample_submission.csv")

random_list = np.random.permutation(len(driver_imgs_list))[:1250]

df_copy = driver_imgs_list.iloc[random_list]

image_paths = [TRAIN_DIR+row.classname+'/'+row.img 

                   for (index, row) in df_copy.iterrows()]

label_list = [int(row.classname[1]) for (index, row) in df_copy.iterrows()]
# One hot vector representation of labels

# y_labels_one_hot = to_categorical(label_list, dtype=np.int8)

x_img_path = np.array(image_paths)



dataset = []

for i in range(len(x_img_path)): # len(x_img_path)

    # load

    img = mpimg.imread(x_img_path[i]) 

    # 此时 img 就已经是一个 np.array 了，可以对它进行任意处理

    dataset.append([img,label_list[i]])

dataset = np.transpose(dataset)
data = np.array(dataset[:][0])

IMG_HEIGHT = 240

IMG_WIDTH = 320



# data = data.reshape(-1,3,IMG_HEIGHT,IMG_WIDTH).transpose([0,2,3,1])

plt.imshow(data[0])
data_gray = [ color.rgb2gray(i) for i in data]

plt.imshow(data_gray[0])
ppc = 16

hog_images = []

hog_features = []

for image in data_gray:

    fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)

    hog_images.append(hog_image)

    hog_features.append(fd)
plt.imshow(hog_images[0])
# labels =  np.array(dataset['labels']).reshape(len(dataset['labels']),1)

labels = np.array(dataset[:][1])
clf = svm.SVC()

hog_features = np.array(hog_features)

data_frame = []

for i in range(len(hog_features)):

    data_frame.append(np.hstack((hog_features[i],labels[i]))) 



np.random.shuffle(data_frame)
#What percentage of data you want to keep for training

percentage = 80

partition = int(len(hog_features)*percentage/100)
data_frame = np.array(data_frame)

x_train, x_test = data_frame[:partition,:-1],  data_frame[partition:,:-1]

y_train, y_test = data_frame[:partition,-1:].ravel() , data_frame[partition:,-1:].ravel()

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy: "+str(accuracy_score(y_test, y_pred)))

print('\n')

print(classification_report(y_test, y_pred))