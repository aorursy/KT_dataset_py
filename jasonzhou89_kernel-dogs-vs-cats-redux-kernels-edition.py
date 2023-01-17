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
#将train目录下训练集数据和test目录下测试集数据读入列表中（文件路径）
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import shutil
import cv2
import sys

train_image_path="../input/train/"
test_image_path="../input/test/"
train_image_list=[]
test_image_list=[]
rows = 299
cols = 299
   
#从目录中所有文件读入到列表中
def get_image_list(path_name, list_name):
    for file_name in os.listdir(path_name):
        list_name.append(os.path.join(path_name, file_name))

get_image_list(train_image_path, train_image_list)
get_image_list(test_image_path, test_image_list)
print("train image sample:{}\ntest image sample:{}".format(len(train_image_list),len(test_image_list)))
#实现显示图片函数，并显示训练集中前10个和后10个图片
def display_img(img_list, summary = True):
    fig = plt.figure(figsize=(15, 3 * math.ceil(len(img_list)/5)))
    for i in range(0, len(img_list)):
        img = cv2.imread(img_list[i])
        img = img[:,:,::-1]#BGR->RGB
        if summary:
            print("---->image: {}  - shape: {}".format(img_list[i], img.shape))
        ax = fig.add_subplot(math.ceil(len(img_list)/5),5,i+1)
        ax.set_title(os.path.basename(img_list[i]))
        ax.set_xticks([])
        ax.set_yticks([])
        img = cv2.resize(img, (128,128))
        ax.imshow(img)
    plt.show()
    
dis_img_list = train_image_list[:10] + train_image_list[-10:]
display_img(dis_img_list)
#导入Xception模型（载入imagenet预训练权值）
from keras.applications import *
model_pre=xception.Xception(weights='imagenet')
#ImageNet 1000个类具体内容，参考文献 https://blog.csdn.net/zhangjunbob/article/details/53258524
#定义猫狗种类
Dogs = [ 'n02085620','n02085782','n02085936','n02086079','n02086240','n02086646','n02086910','n02087046','n02087394','n02088094','n02088238',
        'n02088364','n02088466','n02088632','n02089078','n02089867','n02089973','n02090379','n02090622','n02090721','n02091032','n02091134',
        'n02091244','n02091467','n02091635','n02091831','n02092002','n02092339','n02093256','n02093428','n02093647','n02093754','n02093859',
        'n02093991','n02094114','n02094258','n02094433','n02095314','n02095570','n02095889','n02096051','n02096177','n02096294','n02096437',
        'n02096585','n02097047','n02097130','n02097209','n02097298','n02097474','n02097658','n02098105','n02098286','n02098413','n02099267',
        'n02099429','n02099601','n02099712','n02099849','n02100236','n02100583','n02100735','n02100877','n02101006','n02101388','n02101556',
        'n02102040','n02102177','n02102318','n02102480','n02102973','n02104029','n02104365','n02105056','n02105162','n02105251','n02105412',
        'n02105505','n02105641','n02105855','n02106030','n02106166','n02106382','n02106550','n02106662','n02107142','n02107312','n02107574',
        'n02107683','n02107908','n02108000','n02108089','n02108422','n02108551','n02108915','n02109047','n02109525','n02109961','n02110063',
        'n02110185','n02110341','n02110627','n02110806','n02110958','n02111129','n02111277','n02111500','n02111889','n02112018','n02112137',
        'n02112350','n02112706','n02113023','n02113186','n02113624','n02113712','n02113799','n02113978']
Cats=['n02123045','n02123159','n02123394','n02123597','n02124075','n02125311','n02127052']
#实现判断是否为cat/dog判断函数
def batch_img(img_path_list, batch_size):
    '''split img_path_list into batches'''
    for begin in range(0, len(img_path_list), batch_size):
        end = min(begin+batch_size, len(img_path_list))
        yield img_path_list[begin:end]
        
def read_batch_img(batch_imgpath_list):
    '''read batch img and resize'''
    images = np.zeros((len(batch_imgpath_list), 299, 299, 3), dtype=np.uint8)
    for i in range(len(batch_imgpath_list)):
        img = cv2.imread(batch_imgpath_list[i])
        img = img[:,:,::-1]
        img = cv2.resize(img, (299,299))
        images[i] = img
    return images

def pred_pet(model, img_path_list, top_num, preprocess_input, decode_predictions, batch_size = 32):
    '''predict img
    #returns
        the list, will show pet or not
    '''
    ret = []
    for batch_imgpath_list in batch_img(img_path_list, batch_size):
        X = read_batch_img(batch_imgpath_list)
        X = preprocess_input(X)
        preds = model.predict(X)
        dps = decode_predictions(preds, top = top_num)
        for index in range(len(dps)):
            for i, val in enumerate(dps[index]):
                if (val[0] in Dogs) and ('dog' in batch_imgpath_list[index]):
                    ret.append(True)
                    break
                elif (val[0] in Cats) and ('cat' in batch_imgpath_list[index]):
                    ret.append(True)
                    break
                if i==len(dps[index])-1:
                    ret.append(False)
    return ret     
#显示并剔除异常值
def get_abnormal_v(train_image_list, topN = 50):
    abnormal_v = []
    if os.path.exists("./abnormal.txt"):
        with open("./abnormal.txt", 'r') as f:
            items = f.readlines()
            abnormal_v = [item.strip('\n') for item in items]
    else:
        ret =[]
        ret = pred_pet(model_pre, train_image_list, topN, xception.preprocess_input, xception.decode_predictions)
        for i,val in enumerate(ret):
            if not val:
                abnormal_v.append(train_image_list[i])
        with open("./abnormal.txt", 'w') as f:
            for item in abnormal_v:
                f.write("{}\n".format(item))
    return abnormal_v

abnormal_v = get_abnormal_v(train_image_list, topN=50)
# display_img(abnormal_v, summary = False)
train_image_list = [item for item in train_image_list if item not in abnormal_v]
for i in abnormal_v:
    os.remove(i) 
#shuffle
import random
random.shuffle(train_image_list)

#创建train和val目录，将20%图片作为验证集，80%作为训练集，move猫狗图片
import shutil
def create_dir_and_move_file(image_list):
    dir_list = ["./train/dogs", "./train/cats", "./val/dogs" ,"./val/cats"]
    # 创建多级目录
    for dir_path in dir_list:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)  
    
    # 移动文件或目录
    for file_path in image_list[:math.ceil(len(image_list)/5)]:
        if "dog" in file_path:
            shutil.move(file_path,"./val/dogs")
        else:
            shutil.move(file_path, "./val/cats") 
    for file_path in image_list[math.ceil(len(image_list)/5):]:
        if "dog" in file_path:
            shutil.move(file_path,"./train/dogs")
        else:
            shutil.move(file_path, "./train/cats")
create_dir_and_move_file(train_image_list)
# create the base pre-trained model
base_model = xception.Xception(weights='imagenet', input_shape = (299,299,3), include_top=False, pooling='avg')

x = base_model.output

from keras.models import Model
from keras.layers import Dense
# 二分类分类器
predictions = Dense(1, activation='sigmoid')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
#创建回调函数
from keras import callbacks
class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

history = LossHistory()
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
#设置超参
epochs = 20
batch_size = 32
nb_validation_samples = math.ceil(len(train_image_list)/ 5)
nb_train_samples = 4 * nb_validation_samples
#冻结Xception所有层，值训练top layers
for layer in base_model.layers:
    layer.trainable = False   
#编译模型    
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
#图片数据增强
from keras.preprocessing.image import ImageDataGenerator
#训练数据增强
train_datagen = ImageDataGenerator( preprocessing_function=xception.preprocess_input, 
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True)
#验证数据增强
validation_datagen = ImageDataGenerator(preprocessing_function=xception.preprocess_input)
train_generator = train_datagen.flow_from_directory( "./train/", 
                                                     target_size = (rows, cols),
                                                     batch_size = batch_size,
                                                     class_mode='binary',
                                                     shuffle=True)
validation_generator = validation_datagen.flow_from_directory( "./val/", 
                                                               target_size = (rows, cols),
                                                               batch_size = batch_size, 
                                                               class_mode='binary',
                                                               shuffle=True)
#训练模型
model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(nb_train_samples/batch_size),
                    epochs=epochs, 
                    validation_data=validation_generator, 
                    validation_steps=math.ceil(nb_validation_samples/batch_size),
                    callbacks=[history, earlyStopping])

