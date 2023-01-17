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
import matplotlib.pyplot as plt
import imageio

TEST_FOLDER_PATH="../input/test/"
TRAIN_FOLDER_PATH="../input/train/"
TARGET_NUM=28
label_columns=['Nucleoplasm','Nuclear membrane','Nucleoli',
               'Nucleoli fibrillar center','Nuclear speckles'  ,
               'Nuclear bodies','Endoplasmic reticulum',
               'Golgi apparatus','Peroxisomes'  ,'Endosomes'  ,
               'Lysosomes'  ,'Intermediate filaments'  ,'Actin filaments'  ,
               'Focal adhesion sites'  ,'Microtubules'  ,'Microtubule ends'  ,
               'Cytokinetic bridge'  ,'  Mitotic spindle'  ,
               'Microtubule organizing center'  ,'Centrosome'  ,
               'Lipid droplets'  ,'Plasma membrane'  ,'Cell junctions'  ,
               'Mitochondria'  ,'Aggresome'  ,'Cytosol'  ,'Cytoplasmic bodies'  ,
               'Rods & rings']
label_columns_chinese=['核质','核膜','核仁','核仁纤维中心','核散斑','核机构','内质网',
                       '高尔基体','过氧化物酶体','内体','溶酶体','中间长丝','肌动蛋白丝',
                       '粘着位点','微管','微管末端','细胞动力学桥','有丝分裂纺锤','微管组织中心',
                       '中心体','脂滴','质膜','细胞连接','线粒体','聚集小','细胞质','细胞质体',
                       '杆和环']
label=pd.read_csv('../input/train.csv')
train_file_names=os.listdir(TRAIN_FOLDER_PATH)
test_file_names=os.listdir(TEST_FOLDER_PATH)

TRAIN_SAMPLE_SIZE=label.shape[0]
train_file_names.sort()
test_file_names.sort()
def generate_label_cube(label,label_columns):
    cube=np.zeros((label.shape[0],TARGET_NUM),dtype=int)
    target=label['Target'].values
    for x in range(label.shape[0]):
        for y in target[x].split():
            cube[x][int(y)]=1
    cube=pd.DataFrame(cube,columns=label_columns)
    
    return pd.concat([label,cube],axis=1)
label_new=generate_label_cube(label,label_columns)
tf=imageio.imread("../input/train/"+label[0:1]['Id'].values[0]+'_green.png')
tf.shape
def showFourPic(names):
    fig, axes = plt.subplots(4, 4,figsize=(25,25))
    color=['Blues','Reds','YlOrBr','Greens']
    apex=['_blue','_red','_yellow','_green']
    for x in range(4):
        for y in range(4):
            axes[x,y].imshow(imageio.imread(TRAIN_FOLDER_PATH+names[x]+apex[y]+'.png'),cmap=plt.get_cmap(color[y]))
            axes[x,y].set_title(names[x]+apex[y])
    plt.show()
showFourPic(label_new[label_new['Microtubules']==1]['Id'][0:4].values)
showFourPic(label_new[label_new['Endoplasmic reticulum']==1]['Id'][0:4].values)
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras import layers



def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=name)(x)
    x = Activation('relu', name=name)(x)
    return x
def InceptionV3():

    channel_axis = 3
    classes = 28
    inputs=Input(shape=(512,512,1))

    x = conv2d_bn(inputs,3,3,3,strides=(2,2),padding='valid')
    x = conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    
    # Classification block
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)


    # Create model.
    model = Model(inputs, x, name='inception_v3')

    return model
model = InceptionV3()

model.summary()
def generate_label(target):
    label=np.zeros(TARGET_NUM,dtype=float)
    for x in target.split():
        label[int(x)]=1
    return label

def image_generator(label,batch_size):
    while True:
        try:
            #make batch size data a generator
            batch=label.loc[np.random.randint(TRAIN_SAMPLE_SIZE, size=batch_size)]
            img_list=[]
            img_gen=iter(batch['Id'])
            tar_list=[]
            tar_gen=iter(batch['Target'])
            
            #create train and target batch
            for x in range(batch_size):
                file=next(img_gen)
                target=next(tar_gen)
                img=imageio.imread(TRAIN_FOLDER_PATH+file+'_green.png')
                img_list.append(img)
                tar=generate_label(target)   
                tar_list.append(tar)
                
            #do pre-process and transform
            img_out=np.array(img_list,dtype='float')
            img_out/=255
            img_out=img_out.reshape(batch_size,512,512,1)
            tar_out=np.array(tar_list,dtype='float')

            yield (img_out,tar_out)
        except StopIteration:
            break


predics=model.predict_generator(test_file_generator(test_files),steps=len(test_files))
predics[0]
