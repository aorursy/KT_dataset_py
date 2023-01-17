import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv
import keras
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from keras.applications import vgg19
from keras.models import Sequential,load_model,Model
from keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from tqdm import tqdm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/tianchi-2018-dataset/dataseta_train_20180813/DatasetA_train_20180813"))
def add_colunms(file_path, columns_names, separator):
    data = []
    with open(file_path) as f:
        data = f.read()
        data = data.split('\n')#分行
        data.pop()#去除最后一行空格
        for i in range(len(data)):
            data[i] = data[i].split(separator)
    return pd.DataFrame(data, columns = columns_names)


path_Lab = '../input/tianchi-2018-dataset/dataseta_train_20180813/DatasetA_train_20180813/label_list.txt'
path_Ima = '../input/tianchi-2018-dataset/dataseta_train_20180813/DatasetA_train_20180813/train.txt'
path_APC = '../input/tianchi-2018-dataset/dataseta_train_20180813/DatasetA_train_20180813/attributes_per_class.txt'
path_CWE = '../input/tianchi-2018-dataset/dataseta_train_20180813/DatasetA_train_20180813/class_wordembeddings.txt'

columns_Lab = ['label', 'name']
columns_Ima = ['image', 'label']
columns_APC = ['label', 'animal', 'transportation', 'clothes', 'plant', 'tableware', 'device',
               'black', 'white', 'blue', 'brown', 'orange', 'red', 'green', 'yellow', 'has feathers',
               'has four legs', 'has two legs', 'has two arms', 'for entertainment', 'for business',
               'for communication', 'for family', 'for office use', 'for personal', 'gorgeous',
               'simple', 'elegant', 'cute', 'pure', 'naive']
columns_CWE = ['name']
for i in range(300):
    columns_CWE.append('index{}'.format(i+1))


TrainLab = add_colunms(path_Lab, columns_Lab, '\t')
TrainIma = add_colunms(path_Ima, columns_Ima, '\t')
TrainAPC = add_colunms(path_APC, columns_APC, '\t')
TrainCWE = add_colunms(path_CWE, columns_CWE, ' ')


TrainIma = pd.merge(TrainIma,TrainLab,on='label').drop('label',axis=1)
TrainAPC = pd.merge(TrainLab,TrainAPC,on='label').drop('label',axis=1)
TrainIma_Att = pd.merge(TrainIma,TrainAPC,on='name')
TrainIma_CWE = pd.merge(TrainIma,TrainCWE,on='name')
def GET_Train_Picture(TrainIma_CWE):
    X_train = []
    Y_name = []
    Y_att = []
    for i in range(len(TrainIma_CWE)):
        path = '../input/tianchi-2018-dataset/dataseta_train_20180813/DatasetA_train_20180813/train/' + TrainIma_CWE.iloc[i,0]
        img = image.load_img(path, target_size=(64, 64))
        x = image.img_to_array(img)
        #x = preprocess_input(x,mode='tf')
        X_train.append(x)
        Y_name.append(TrainIma_CWE.iloc[i,1])
        Y_att.append(np.array(TrainIma_CWE.iloc[i,2:302], np.float32))
    return np.array(X_train), np.array(Y_name), np.array(Y_att)
X_train, Y_name, Y_train = GET_Train_Picture(TrainIma_CWE)
print(X_train.shape, Y_name.shape, Y_train.shape)
model = load_model('../input/class-wordembeddings/model_VGG19_CWE30.h5')
model.compile(loss='mean_squared_error',optimizer='Adam',metrics=['mae'])
model.summary()
output = model.predict(X_train)
def calEuclideanDistance(vec1,vec2):  
    dist = np.sqrt(np.sum((vec1 - vec2)*(vec1 - vec2), axis = 1))
    return dist
#Top10
def AttSimilarity(predict,Information):
    outputList = []
    ClassifierWeight = np.array(Information.iloc[:, 1:301], np.float32)
    for k in tqdm(range(len(predict))):
        ClassifierName = pd.DataFrame({'name':Information.name, 'distance': None})
        distance = calEuclideanDistance(ClassifierWeight, predict[k])
        ClassifierName.loc[:,'distance'] = distance.reshape([-1,1]) 
        ClassifierName = ClassifierName.sort_values(by='distance')
        bestfit = ClassifierName.iloc[0:10,1].tolist()
        outputList.append(bestfit)
    return outputList
allinhere = AttSimilarity(output, TrainCWE)

CompareList = pd.DataFrame({'res':allinhere,'truth':TrainIma_CWE.name})
def compare(a,b):
    if b in a:
        return 1
    else:
        return 0
CompareList['score'] = CompareList.apply(lambda x: compare(x.res,x.truth),axis =1)
print('Score:', (CompareList.score.sum(axis = 0)*100 / len(CompareList.score) ),'%')
TestIma.result = allinhere
f = open('result.txt','w')
for i in range(len(TestIma)): 
    f.write(TestIma.image[i]+'\t'+TestIma.result[i]+'\n')
f.close()