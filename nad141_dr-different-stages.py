# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import glob

import cv2

import matplotlib.pyplot as plt

import random as r

%matplotlib inline
train = pd.read_csv('../input/train.csv')
path_train = glob.glob('../input/train_images/*')
index = 5
file_name = path_train[index].split('/')[-1]
# img = cv2.imread(path_train[index],0)

img = cv2.imread('../input/train_images/0083ee8054ee.png',0)



img.shape
row, col = img.shape

nr,nc = cv2.getOptimalDFTSize(row), cv2.getOptimalDFTSize(col)

print(nr,', ',nc)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft)



magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))



plt.subplot(121),plt.imshow(img)

# plt.title(file_name + str(train[train['id_code'] == file_name.split('.')[0]].diagnosis.values)), plt.xticks([]), plt.yticks([])

plt.title('0083ee8054ee' + str(train[train['id_code'] == '0083ee8054ee'].diagnosis.values)), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(magnitude_spectrum)

plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

plt.show()
def show_image(image):

    plt.imshow(image.image_name)

    plt.title(image.type), plt.xticks([]), plt.yticks([])
class ImageDR(object):

    def __init__(self, name, dr_type, image_array):

        self.image_array = image_array

        self.name = name

        self.dr_type = dr_type

    def get_plot(self):

        plt.imshow(self.image_array)

        plt.title(self.name + ':' + dr_dict[self.dr_type]),plt.xticks([]),plt.yticks([])

        plt.show()

    def get_image(self):

        return self.image_array

    def __str__(self):

        return self.name + ':' + dr_dict[self.dr_type]
train.head()
train.diagnosis.value_counts().plot(kind ='barh')
dr_dict = {'0': 'NO_DR', '1': 'mild', '2': 'moderate', '3': 'severe', '4': 'proliferative'}
r.seed(42)
train_4 = train[train['diagnosis'] == 4]
train_4_10 = train_4.iloc[r.sample(range(train_4.shape[0]), 20)]
train_4_10 = train_4_10.id_code.values
train_4_10
img = cv2.imread('../input/train_images/' + str(train_4_10[13]) + '.png')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imgDR_4 = ImageDR(train_4_10[7], '4', img)
imgDR_4.get_plot()
for i in train_4_10:

    img = cv2.imread('../input/train_images/' + str(i) +'.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgDR = ImageDR(i,'4', img)

    imgDR.get_plot()
train_3 = train[train['diagnosis'] == 3]

train_3_10 = train_3.iloc[r.sample(range(train_3.shape[0]), 10)]

train_3_10 = train_3_10.id_code.values
img2 = cv2.imread('../input/train_images/' + str(train_3_10[2]) + '.png')

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
imgDR_3 = ImageDR(train_3_10[2], '3', img2)
imgDR_3.get_plot()
for i in train_3_10:

    img = cv2.imread('../input/train_images/' + str(i) +'.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgDR = ImageDR(i,'3', img)

    imgDR.get_plot()
train_2 = train[train['diagnosis'] == 2]

train_2_10 = train_2.iloc[r.sample(range(train_2.shape[0]), 10)]

train_2_10 = train_2_10.id_code.values
img3 = cv2.imread('../input/train_images/' + str(train_2_10[2]) + '.png')

img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
imgDR_2 = ImageDR(train_2_10[2], '2', img3)
imgDR_2.get_plot()
for i in train_2_10:

    img = cv2.imread('../input/train_images/' + str(i) +'.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgDR = ImageDR(i,'2', img)

    imgDR.get_plot()
train_1 = train[train['diagnosis'] == 1]

train_1_10 = train_1.iloc[r.sample(range(train_1.shape[0]), 10)]

train_1_10 = train_1_10.id_code.values
img4 = cv2.imread('../input/train_images/' + str(train_1_10[2]) + '.png')

img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
imgDR_1 = ImageDR(train_1_10[2], '1', img4)
imgDR_1.get_plot()
for i in train_1_10:

    img = cv2.imread('../input/train_images/' + str(i) +'.png')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imgDR = ImageDR(i,'1', img)

    imgDR.get_plot()