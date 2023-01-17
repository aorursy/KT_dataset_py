import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras.preprocessing.image import ImageDataGenerator
def getFileNameData(array_,path):
    for name in glob.glob(path+'/***'):
        array_.append(name)
#         array_ = np.append(array_,name)
# LabelledRice  RiceDiseaseDataset
# !ls '../input/rice-diseases-image-dataset/RiceDiseaseDataset/train'
path_train = '../input/rice-diseases-image-dataset/LabelledRice/Labelled'
path_validation = '../input/rice-diseases-image-dataset/RiceDiseaseDataset/validation'
# GetPath Data Train
# fileTrain = []
# getFileNameData(fileTrain,path_train)
# print(fileTrain)

label_path = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/BrownSpot'
label_path_leaf = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/LeafBlast'
label_path_hispa = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/Hispa'
label_path_health = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/Healthy'

# GetFile Data Train
hispa = []
leafblast = []
brownspot = []
healthy = []

getFileNameData(leafblast,label_path_leaf)
getFileNameData(hispa,label_path_hispa)
getFileNameData(healthy,label_path_health)
getFileNameData(brownspot,label_path)

# getFileNameData(hispa,fileTrain[0])
# getFileNameData(leafblast,fileTrain[1])
# getFileNameData(healthy,fileTrain[3])
# getFileNameData(brownspot,fileTrain[3])
print(len(hispa),len(leafblast),len(brownspot),len(healthy))

hispa = np.array(hispa)
leafblast = np.array(leafblast)
brownspot = np.array(brownspot)
healthy = np.array(healthy)
print(hispa.shape,leafblast.shape,brownspot.shape,healthy.shape)
print(fileTrain[0],fileTrain[1],fileTrain[2],fileTrain[3])
fig = plt.figure()
a = fig.add_subplot(2, 3, 1)
imgplot = plt.imshow(cv2.imread('../input/rice-diseases-image-dataset/RiceDiseaseDataset/LabelledRice/Labelled/BrownSpot/'))
a.set_title('Hispa')
a = fig.add_subplot(2, 3, 2)
imgplot = plt.imshow(cv2.imread(dataTrain[1][0]))
imgplot.set_clim(0.0, 0.7)
a.set_title('LeafBlasr')
a = fig.add_subplot(2, 3, 3)
imgplot = plt.imshow(cv2.imread(dataTrain[2][0]))
a.set_title('BrownSpot')
a = fig.add_subplot(2, 3, 4)
imgplot = plt.imshow(cv2.imread(dataTrain[2][0]))
a.set_title('Healthy')
def resize_img(array_,result):
    for i in array_:
        img = cv2.imread(i)
        scale_percent = 80 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        result.append(cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA))
def rotate_img(array_,result):
    for i in array_:
        img = cv2.imread(i)
        scale_percent = 80 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        result.append(cv2.resize(img_rotate_90_clockwise, (224,224), interpolation = cv2.INTER_AREA))
resize_hispa,resize_leafblast,resize_brownspot,resize_healthy = [],[],[],[]
rotate_hispa,rotate_leafblast,rotate_brownspot,rotate_healthy = [],[],[],[]

resize_img(hispa,resize_hispa)
resize_img(leafblast,resize_leafblast)
resize_img(brownspot,resize_brownspot)
resize_img(healthy,resize_healthy)

rotate_img(hispa,rotate_hispa)
rotate_img(leafblast,rotate_leafblast)
rotate_img(brownspot,rotate_brownspot)
rotate_img(healthy,rotate_healthy)

len(resize_hispa),len(resize_leafblast),len(resize_brownspot),len(resize_healthy),len(rotate_hispa),len(rotate_leafblast),len(rotate_brownspot),len(rotate_healthy)
resize_hispa,resize_leafblast,resize_brownspot,resize_healthy = np.array(resize_hispa),np.array(resize_leafblast),np.array(resize_brownspot),np.array(resize_healthy)
rotate_hispa,rotate_leafblast,rotate_brownspot,rotate_healthy = np.array(rotate_hispa),np.array(rotate_leafblast),np.array(rotate_brownspot),np.array(rotate_healthy)
# resize_healthy = np.array(resize_healthy)
resize_healthy.shape,rotate_healthy.shape
# fig = plt.figure()
# a = fig.add_subplot(2, 3, 1)
# imgplot = plt.imshow(resize_hispa[0])
# a.set_title('Hispa')
# a = fig.add_subplot(2, 3, 2)
# imgplot = plt.imshow(rotate_hispa[0])
# imgplot.set_clim(0.0, 0.7)
# a.set_title('Rotate Hispa')
# a = fig.add_subplot(2, 3, 3)
# imgplot = plt.imshow(cv2.imread(dataTrain[2][0]))
# a.set_title('BrownSpot')
# a = fig.add_subplot(2, 3, 4)
# imgplot = plt.imshow(cv2.imread(dataTrain[2][0]))
# a.set_title('Healthy')
hispas = np.concatenate((resize_hispa,rotate_hispa))
leafblasts = np.concatenate((resize_leafblast,rotate_leafblast))
brownspots = np.concatenate((resize_brownspot,rotate_brownspot))
healthys = np.concatenate((resize_healthy,rotate_healthy))
hispas.shape,healthys.shape,brownspots.shape,leafblasts.shape
# data_train_resize = np.concatenate((resize_hispa,resize_leafblast,resize_brownspot,resize_healthy))
# data_train_resize.shape
data_train_resize = np.concatenate((hispas,leafblasts,brownspots,healthys))
data_train_resize.shape
!ls '../working/'
np.save('data_train_resize_rotate_224.npy',data_train_resize)
# np.save('data_train_resize_leafblast.npy',resize_leafblast)
# np.save('data_train_resize_brownspot.npy',resize_brownspot)
# np.save('data_train_resize_healthy.npy',resize_healthy)
from IPython.display import FileLink
FileLink(r'data_train_resize_rotate_224.npy')
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
label_path = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/***'
label_list = []
for name in glob.glob(label_path):
    a = name.replace('../input/rice-diseases-image-dataset/LabelledRice/Labelled/','')
    for i in glob.glob(name+'/***.jpg'):
        label_list.append(a)
def getData(data,array_,value):
    for name in glob.glob(data):
        a = name.replace('../input/rice-diseases-image-dataset/LabelledRice/Labelled/','')
        for i in glob.glob(name+'/***.jpg'):
            array_.append(value)
label_path = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/BrownSpot'
label_path_leaf = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/LeafBlast'
label_path_hispa = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/Hispa'
label_path_health = '../input/rice-diseases-image-dataset/LabelledRice/Labelled/Healthy'
brownspot = []
hispa = []
healthy = []
leafblast = []

getData(label_path,brownspot,[0,0,1,0])
getData(label_path_leaf,leafblast,[0,1,0,0])
getData(label_path_hispa,hispa,[1,0,0,0])
getData(label_path_health,healthy,[0,0,0,1])

len(brownspot), len(hispa), len(healthy), len(leafblast)
dup_brownspot = brownspot
dup_hispa = hispa
dup_healthy = healthy
dup_leafblast = leafblast
len(dup_brownspot),len(dup_hispa),len(dup_healthy),len(dup_leafblast)
# labelClass = np.concatenate((np.array(hispa),np.array(leafblast),np.array(brownspot),np.array(healthy)))
# labelClass = np.concatenate(hispa,leafblast,brownspot,healthy)
labelClass = [*hispa,*dup_hispa,*leafblast,*dup_leafblast,*brownspot,*dup_brownspot,*healthy,*dup_healthy]
hispa[0],leafblast[0],brownspot[0],healthy[0],len(labelClass)
label_binarizer = LabelBinarizer()
image_labels = label_binarizer.fit_transform(label_list)
pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
n_classes = len(label_binarizer.classes_)
n_classes
print(label_binarizer.classes_,len(image_labels))
np.save('image_labels_resize_rotate.npy',labelClass)
from IPython.display import FileLink
FileLink(r'image_labels_resize_rotate.npy')