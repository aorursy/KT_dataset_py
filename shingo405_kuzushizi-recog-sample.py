!pip install japanize-matplotlib

!pip install cnn_finetune
import matplotlib

import pandas as pd

import numpy as np

from PIL import Image

from pylab import rcParams

import os

import json

import matplotlib

import matplotlib.pyplot as plt

import glob

import japanize_matplotlib



%matplotlib inline
!unzip ../input/kuzushiji-recognition/train_images.zip -d train_images >/dev/null

!unzip ../input/kuzushiji-recognition/test_images.zip -d test_images >/dev/null
ut=pd.read_csv("../input/kuzushiji-recognition/unicode_translation.csv")

ut_dict=ut.set_index("Unicode")["char"].to_dict()

train=pd.read_csv("../input/kuzushiji-recognition/train.csv")

train_image_id=[os.path.basename(p).split(".")[0] for p in glob.glob("./train_images/*.jpg")]

train=train[train["image_id"].isin(train_image_id)]
lists=[]

for image_id,labels in train.values:

    if labels == labels:

        df=pd.DataFrame([],columns=["image_id","label"])

        df["label"]=[label for i,label in enumerate(labels.split(" ")) if i%5==0]

        df["X"]=[int(label) for i,label in enumerate(labels.split(" ")) if i%5==1]

        df["Y"]=[int(label) for i,label in enumerate(labels.split(" ")) if i%5==2]

        df["width"]=[int(label) for i,label in enumerate(labels.split(" ")) if i%5==3]

        df["height"]=[int(label) for i,label in enumerate(labels.split(" ")) if i%5==4]

        df["image_id"]=image_id

        lists.append(df)

train_labels=pd.concat(lists,ignore_index=True) 
train_labels.head(5)
# 出現文字数

train_labels["label"].map(ut_dict).value_counts()[:20]
categories = [str(i) for i in train_labels["label"]]

unicode_categories = [ut_dict[i] if i in ut_dict.keys() else "-" for i in categories]

label2id={l:i for i,l in enumerate(train_labels["label"])}
def image_write(i,bboxes_df,folder="train_images",label_show=True):

    image_id=bboxes_df["image_id"].unique()[i]

    image_name="./"+folder+"/"+image_id+".jpg"

    img=Image.open(image_name)

    num_img=np.array(img)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.imshow(img)

    for image_id,label,X,Y,width,height in bboxes_df[["image_id","label","X","Y","width","height"]].query("image_id=='{j}'".format(j=image_id)).values:

        rect = plt.Rectangle((X,Y),width,height,color="red",fill=False)

        ax.add_patch(rect)

        if label_show:

            ax.text(X+width, Y+height/2, ut_dict[label], size = 16, color = "blue")

    plt.figure(figsize=(50,50))

    plt.show()
img_num=1



rcParams['figure.figsize']=[10,10]

image_write(img_num,train_labels)
import numpy as np

import json

import pandas as pd

from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

from pandas.io.json import json_normalize

import random

import tensorflow as tf

from sklearn.utils import shuffle

from sklearn.model_selection import KFold,train_test_split

import matplotlib.pyplot as plt

import glob

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Dense,Dropout, Conv2D,Conv2DTranspose, BatchNormalization, Activation,AveragePooling2D,GlobalAveragePooling2D, Input, Concatenate, MaxPool2D, Add, UpSampling2D, LeakyReLU,ZeroPadding2D

from keras.models import Model

from keras.objectives import mean_squared_error

from keras import backend as K

from keras.losses import binary_crossentropy

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler

import os  

import keras

from keras.optimizers import Adam, RMSprop, SGD

from tensorflow.compat.v1 import ConfigProto

from tensorflow.compat.v1 import InteractiveSession
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) > 0:

    for k in range(len(physical_devices)):

        tf.config.experimental.set_memory_growth(physical_devices[k], True)

        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))

else:

    print("Not enough GPU hardware devices available")
path_1="../input/kuzushiji-recognition/train.csv"

path_2="./train_images/"

path_3="./test_images/"

path_4="../input/kuzushiji-recognition/sample_submission.csv"

df_train=pd.read_csv(path_1)



train_image_id=[os.path.basename(p).split(".")[0] for p in glob.glob(path_2+"*.jpg")]

df_train=df_train[df_train["image_id"].isin(train_image_id)]



#print(df_train.head())

#print(df_train.shape)

df_train=df_train.dropna(axis=0, how='any')#you can use nan data(page with no letter)

df_train=df_train.reset_index(drop=True)

#print(df_train.shape)



annotation_list_train=[]

category_names=set()



for i in range(len(df_train)):

    ann=np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,x,y,width,height for each picture

    category_names=category_names.union({i for i in ann[:,0]})



category_names=sorted(category_names)

dict_cat={list(category_names)[j]:str(j) for j in range(len(category_names))}

inv_dict_cat={str(j):list(category_names)[j] for j in range(len(category_names))}

#print(dict_cat)

  

for i in range(len(df_train)):

    ann=np.array(df_train.loc[i,"labels"].split(" ")).reshape(-1,5)#cat,left,top,width,height for each picture

    for j,category_name in enumerate(ann[:,0]):

        ann[j,0]=int(dict_cat[category_name])  

    ann=ann.astype('int32')

    ann[:,1]+=ann[:,3]//2#center_x

    ann[:,2]+=ann[:,4]//2#center_y

    annotation_list_train.append(["{}{}.jpg".format(path_2,df_train.loc[i,"image_id"]),ann])

# get directory of test images

df_submission=pd.read_csv(path_4).reset_index(drop=True)



test_image_id=[os.path.basename(p).split(".")[0] for p in glob.glob(path_3+"*.jpg")]

df_submission=df_submission[df_submission["image_id"].isin(test_image_id)]



id_test=path_3+df_submission["image_id"].values+".jpg"
aspect_ratio_pic_all=[]

aspect_ratio_pic_all_test=[]

average_letter_size_all=[]

train_input_for_size_estimate=[]

for i in range(len(annotation_list_train)):

    with Image.open(annotation_list_train[i][0]) as f:

        width,height=f.size

        area=width*height

        aspect_ratio_pic=height/width

        aspect_ratio_pic_all.append(aspect_ratio_pic)

        letter_size=annotation_list_train[i][1][:,3]*annotation_list_train[i][1][:,4]

        letter_size_ratio=letter_size/area

    

        average_letter_size=np.mean(letter_size_ratio)

        average_letter_size_all.append(average_letter_size)

        train_input_for_size_estimate.append([annotation_list_train[i][0],np.log(average_letter_size)])#logにしとく

    



for i in range(len(id_test)):

    with Image.open(id_test[i]) as f:

        width,height=f.size

        aspect_ratio_pic=height/width

        aspect_ratio_pic_all_test.append(aspect_ratio_pic)



rcParams['figure.figsize']=[6,6]

plt.hist(np.log(average_letter_size_all),bins=100)

plt.title('log(ratio of letter_size to picture_size))',loc='center',fontsize=12)

plt.show()


category_n=1

import cv2

input_width,input_height=512, 512



def Datagen_sizecheck_model(filenames, batch_size, size_detection_mode=True, is_train=True,random_crop=True):

    x=[]

    y=[]

    

    count=0



    while True:

        for i in range(len(filenames)):

            if random_crop:

                crop_ratio=np.random.uniform(0.7,1)

            else:

                crop_ratio=1

            with Image.open(filenames[i][0]) as f:

                #random crop

                if random_crop and is_train:

                    pic_width,pic_height=f.size

                    f=np.asarray(f.convert('RGB'),dtype=np.uint8)

                    top_offset=np.random.randint(0,pic_height-int(crop_ratio*pic_height))

                    left_offset=np.random.randint(0,pic_width-int(crop_ratio*pic_width))

                    bottom_offset=top_offset+int(crop_ratio*pic_height)

                    right_offset=left_offset+int(crop_ratio*pic_width)

                    f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))

                else:

                    f=f.resize((input_width, input_height))

                    f=np.asarray(f.convert('RGB'),dtype=np.uint8)                    

                x.append(f)

            

            

            if random_crop and is_train:

                y.append(filenames[i][1]-np.log(crop_ratio))

            else:

                y.append(filenames[i][1])

            

            count+=1

            if count==batch_size:

                x=np.array(x, dtype=np.float32)

                y=np.array(y, dtype=np.float32)



                inputs=x/255

                targets=y             

                x=[]

                y=[]

                count=0

                yield inputs, targets







def aggregation_block(x_shallow, x_deep, deep_ch, out_ch):

    x_deep= Conv2DTranspose(deep_ch, kernel_size=2, strides=2, padding='same', use_bias=False)(x_deep)

    x_deep = BatchNormalization()(x_deep)     

    x_deep = LeakyReLU(alpha=0.1)(x_deep)

    x = Concatenate()([x_shallow, x_deep])

    x=Conv2D(out_ch, kernel_size=1, strides=1, padding="same")(x)

    x = BatchNormalization()(x)     

    x = LeakyReLU(alpha=0.1)(x)

    return x

    





def cbr(x, out_layer, kernel, stride):

    x=Conv2D(out_layer, kernel_size=kernel, strides=stride, padding="same")(x)

    x = BatchNormalization()(x)

    x = LeakyReLU(alpha=0.1)(x)

    return x



def resblock(x_in,layer_n):

    x=cbr(x_in,layer_n,3,1)

    x=cbr(x,layer_n,3,1)

    x=Add()([x,x_in])

    return x    





#I use the same network at CenterNet

def create_model(input_shape, size_detection_mode=True, aggregation=True):

        input_layer = Input(input_shape)

        

        #resized input

        input_layer_1=AveragePooling2D(2)(input_layer)

        input_layer_2=AveragePooling2D(2)(input_layer_1)



        #### ENCODER ####



        x_0= cbr(input_layer, 16, 3, 2)#512->256

        concat_1 = Concatenate()([x_0, input_layer_1])



        x_1= cbr(concat_1, 32, 3, 2)#256->128

        concat_2 = Concatenate()([x_1, input_layer_2])



        x_2= cbr(concat_2, 64, 3, 2)#128->64

        

        x=cbr(x_2,64,3,1)

        x=resblock(x,64)

        x=resblock(x,64)

        

        x_3= cbr(x, 128, 3, 2)#64->32

        x= cbr(x_3, 128, 3, 1)

        x=resblock(x,128)

        x=resblock(x,128)

        x=resblock(x,128)

        

        x_4= cbr(x, 256, 3, 2)#32->16

        x= cbr(x_4, 256, 3, 1)

        x=resblock(x,256)

        x=resblock(x,256)

        x=resblock(x,256)

        x=resblock(x,256)

        x=resblock(x,256)

 

        x_5= cbr(x, 512, 3, 2)#16->8

        x= cbr(x_5, 512, 3, 1)

        

        x=resblock(x,512)

        x=resblock(x,512)

        x=resblock(x,512)

        

        if size_detection_mode:

            x=GlobalAveragePooling2D()(x)

            x=Dropout(0.2)(x)

            out=Dense(1,activation="linear")(x)

        

        else:#centernet mode

        #### DECODER ####

            x_1= cbr(x_1, output_layer_n, 1, 1)

            x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

            x_2= cbr(x_2, output_layer_n, 1, 1)

            x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)

            x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

            x_3= cbr(x_3, output_layer_n, 1, 1)

            x_3 = aggregation_block(x_3, x_4, output_layer_n, output_layer_n) 

            x_2 = aggregation_block(x_2, x_3, output_layer_n, output_layer_n)

            x_1 = aggregation_block(x_1, x_2, output_layer_n, output_layer_n)

            

            x_4= cbr(x_4, output_layer_n, 1, 1)



            x=cbr(x, output_layer_n, 1, 1)

            x= UpSampling2D(size=(2, 2))(x)#8->16 tconvのがいいか



            x = Concatenate()([x, x_4])

            x=cbr(x, output_layer_n, 3, 1)

            x= UpSampling2D(size=(2, 2))(x)#16->32

        

            x = Concatenate()([x, x_3])

            x=cbr(x, output_layer_n, 3, 1)

            x= UpSampling2D(size=(2, 2))(x)#32->64     128のがいいかも？ 

        

            x = Concatenate()([x, x_2])

            x=cbr(x, output_layer_n, 3, 1)

            x= UpSampling2D(size=(2, 2))(x)#64->128 

            

            x = Concatenate()([x, x_1])

            x=Conv2D(output_layer_n, kernel_size=3, strides=1, padding="same")(x)

            out = Activation("sigmoid")(x)

        

        model=Model(input_layer, out)

        

        return model

    

        





def model_fit_sizecheck_model(model,train_list,cv_list,n_epoch,batch_size=32):

        hist = model.fit_generator(

                Datagen_sizecheck_model(train_list,batch_size, is_train=True,random_crop=True),

                steps_per_epoch = len(train_list) // batch_size,

                epochs = n_epoch,

                validation_data=Datagen_sizecheck_model(cv_list,batch_size, is_train=False,random_crop=False),

                validation_steps = len(cv_list) // batch_size,

                callbacks = [lr_schedule, model_checkpoint],#[early_stopping, reduce_lr, model_checkpoint],

                shuffle = True,

                verbose = 1

        )

        return hist



    

if not os.path.exists("./models"):

    os.makedirs("./models")
K.clear_session()

model=create_model(input_shape=(input_height,input_width,3))



"""

# EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 10, verbose = 1)

# ModelCheckpoint

weights_dir = '/model_1/'

if os.path.exists(weights_dir) == False:os.mkdir(weights_dir)

model_checkpoint = ModelCheckpoint(weights_dir + "val_loss{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,

                                      save_best_only = True, save_weights_only = True, period = 1)

# reduce learning rate

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)

"""

def lrs(epoch):

    lr = 0.0001

    return lr



lr_schedule = LearningRateScheduler(lrs)

model_checkpoint = ModelCheckpoint("./models/centernet_step1.hdf5", monitor = 'val_loss', verbose = 1,

                                      save_best_only = True, save_weights_only = True, period = 1)
## step1 training

train_list, cv_list = train_test_split(train_input_for_size_estimate, random_state = 111,test_size = 0.2)

# for layer in model.layers:

#     layer.trainable = False



learning_rate=0.0005

n_epoch=0

batch_size=4



model.compile(loss=mean_squared_error, optimizer=Adam(lr=learning_rate))

hist = model_fit_sizecheck_model(model,train_list,cv_list,n_epoch,batch_size)



#model.save_weights('./centernet_step1.h5')
## step1 predict

#model.load_weights('./models/centernet_step1.h5')

model.load_weights('../input/models/centernet_step1.h5')

predict = model.predict_generator(Datagen_sizecheck_model(cv_list,batch_size, is_train=False,random_crop=False),

                                  steps=len(cv_list) // batch_size)

target=[cv[1] for cv in cv_list]

plt.scatter(predict,target[:len(predict)])

plt.title('---letter_size/picture_size--- estimated vs target ',loc='center',fontsize=10)

plt.show()
batch_size=1

predict_train = model.predict_generator(Datagen_sizecheck_model(train_input_for_size_estimate,batch_size, is_train=False,random_crop=False, ),

                                  steps=len(train_input_for_size_estimate)//batch_size,verbose=1)
base_detect_num_h,base_detect_num_w=25,25

annotation_list_train_w_split=[]

for i, predicted_size in enumerate(predict_train):

    detect_num_h=aspect_ratio_pic_all[i]*np.exp(-predicted_size/2)

    detect_num_w=detect_num_h/aspect_ratio_pic_all[i]

    h_split_recommend=np.maximum(1,detect_num_h/base_detect_num_h)

    w_split_recommend=np.maximum(1,detect_num_w/base_detect_num_w)

    annotation_list_train_w_split.append([annotation_list_train[i][0],annotation_list_train[i][1],h_split_recommend,w_split_recommend])

for i in np.arange(0,3):

    print("recommended height split:{}, recommended width_split:{}".format(annotation_list_train_w_split[i][2],annotation_list_train_w_split[i][3]))

    img = np.asarray(Image.open(annotation_list_train_w_split[i][0]).convert('RGB'))

    plt.imshow(img)

    plt.show()
predict_train
category_n=1

output_layer_n=category_n+4

output_height,output_width=128,128



def Datagen_centernet(filenames, batch_size):

    x=[]

    y=[]

    

    count=0



    while True:

        for i in range(len(filenames)):

            h_split=filenames[i][2]

            w_split=filenames[i][3]

            max_crop_ratio_h=1/h_split

            max_crop_ratio_w=1/w_split

            crop_ratio=np.random.uniform(0.5,1)

            crop_ratio_h=max_crop_ratio_h*crop_ratio

            crop_ratio_w=max_crop_ratio_w*crop_ratio

            

            with Image.open(filenames[i][0]) as f:

                

                #random crop

                

                pic_width,pic_height=f.size

                f=np.asarray(f.convert('RGB'),dtype=np.uint8)

                top_offset=np.random.randint(0,pic_height-int(crop_ratio_h*pic_height))

                left_offset=np.random.randint(0,pic_width-int(crop_ratio_w*pic_width))

                bottom_offset=top_offset+int(crop_ratio_h*pic_height)

                right_offset=left_offset+int(crop_ratio_w*pic_width)

                f=cv2.resize(f[top_offset:bottom_offset,left_offset:right_offset,:],(input_height,input_width))

                x.append(f)            



            output_layer=np.zeros((output_height,output_width,(output_layer_n+category_n)))

            for annotation in filenames[i][1]:

                x_c=(annotation[1]-left_offset)*(output_width/int(crop_ratio_w*pic_width))

                y_c=(annotation[2]-top_offset)*(output_height/int(crop_ratio_h*pic_height))

                width=annotation[3]*(output_width/int(crop_ratio_w*pic_width))

                height=annotation[4]*(output_height/int(crop_ratio_h*pic_height))

                top=np.maximum(0,y_c-height/2)

                left=np.maximum(0,x_c-width/2)

                bottom=np.minimum(output_height,y_c+height/2)

                right=np.minimum(output_width,x_c+width/2)

                    

                if top>=(output_height-0.1) or left>=(output_width-0.1) or bottom<=0.1 or right<=0.1:#random crop(out of picture)

                    continue

                width=right-left

                height=bottom-top

                x_c=(right+left)/2

                y_c=(top+bottom)/2



                

                category=0#not classify, just detect

                heatmap=((np.exp(-(((np.arange(output_width)-x_c)/(width/10))**2)/2)).reshape(1,-1)

                                                        *(np.exp(-(((np.arange(output_height)-y_c)/(height/10))**2)/2)).reshape(-1,1))

                output_layer[:,:,category]=np.maximum(output_layer[:,:,category],heatmap[:,:])

                output_layer[int(y_c//1),int(x_c//1),category_n+category]=1

                output_layer[int(y_c//1),int(x_c//1),2*category_n]=y_c%1#height offset

                output_layer[int(y_c//1),int(x_c//1),2*category_n+1]=x_c%1

                output_layer[int(y_c//1),int(x_c//1),2*category_n+2]=height/output_height

                output_layer[int(y_c//1),int(x_c//1),2*category_n+3]=width/output_width

            y.append(output_layer)    

        

            count+=1

            if count==batch_size:

                x=np.array(x, dtype=np.float32)

                y=np.array(y, dtype=np.float32)



                inputs=x/255

                targets=y             

                x=[]

                y=[]

                count=0

                yield inputs, targets



def all_loss(y_true, y_pred):

        mask=K.sign(y_true[...,2*category_n+2])

        N=K.sum(mask)

        alpha=2.

        beta=4.



        heatmap_true_rate = K.flatten(y_true[...,:category_n])

        heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])

        heatmap_pred = K.flatten(y_pred[...,:category_n])

        heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))

        offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))

        sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))

        

        all_loss=(heatloss+1.0*offsetloss+5.0*sizeloss)/N

        return all_loss



def size_loss(y_true, y_pred):

        mask=K.sign(y_true[...,2*category_n+2])

        N=K.sum(mask)

        sizeloss=K.sum(K.abs(y_true[...,2*category_n+2]-y_pred[...,category_n+2]*mask)+K.abs(y_true[...,2*category_n+3]-y_pred[...,category_n+3]*mask))

        return (5*sizeloss)/N



def offset_loss(y_true, y_pred):

        mask=K.sign(y_true[...,2*category_n+2])

        N=K.sum(mask)

        offsetloss=K.sum(K.abs(y_true[...,2*category_n]-y_pred[...,category_n]*mask)+K.abs(y_true[...,2*category_n+1]-y_pred[...,category_n+1]*mask))

        return (offsetloss)/N

    

def heatmap_loss(y_true, y_pred):

        mask=K.sign(y_true[...,2*category_n+2])

        N=K.sum(mask)

        alpha=2.

        beta=4.



        heatmap_true_rate = K.flatten(y_true[...,:category_n])

        heatmap_true = K.flatten(y_true[...,category_n:(2*category_n)])

        heatmap_pred = K.flatten(y_pred[...,:category_n])

        heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))

        return heatloss/N



    

def model_fit_centernet(model,train_list,cv_list,n_epoch,batch_size=32):

        hist = model.fit_generator(

                Datagen_centernet(train_list,batch_size),

                steps_per_epoch = len(train_list) // batch_size,

                epochs = n_epoch,

                validation_data=Datagen_centernet(cv_list,batch_size),

                validation_steps = len(cv_list) // batch_size,

                callbacks = [lr_schedule],#early_stopping, reduce_lr, model_checkpoint],

                shuffle = True,

                verbose = 1

        )

        return hist
import keras

K.clear_session()

model=create_model(input_shape=(input_height,input_width,3),size_detection_mode=False)



def lrs(epoch):

    lr = 0.0005

    if epoch >= 20: lr = 0.0002

    return lr



lr_schedule = LearningRateScheduler(lrs)



# EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 60, verbose = 1)

model_checkpoint = ModelCheckpoint("./models/val_loss{val_loss:.3f}.hdf5", monitor = 'val_loss', verbose = 1,

                                      save_best_only = True, save_weights_only = True, period = 3)

# reduce learning rate

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 10, verbose = 1)

model.load_weights('../input/models/centernet_step1.h5',by_name=True, skip_mismatch=True)

# model.load_weights('./models/centernet_step1.h5',by_name=True, skip_mismatch=True)
train_list, cv_list = train_test_split(annotation_list_train_w_split, random_state = 111,test_size = 0.2)#stratified split is better

n_epoch=0

batch_size=1

model.compile(loss=all_loss, optimizer=Adam(lr=learning_rate), metrics=[heatmap_loss,size_loss,offset_loss])

hist = model_fit_centernet(model,train_list,cv_list,n_epoch,batch_size)



#model.save_weights('./models/centernet_step2.h5')
# predict(一部)



#model.load_weights('./models/centernet_step2.h5')

model.load_weights('../input/models/centernet_step2.h5')

pred_in_h=512

pred_in_w=512

pred_out_h=int(pred_in_h/4)

pred_out_w=int(pred_in_w/4)



for i in np.arange(0,1):

    img = np.asarray(Image.open(cv_list[i][0]).resize((pred_in_w,pred_in_h)).convert('RGB'))

    predict=model.predict((img.reshape(1,pred_in_h,pred_in_w,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))

    heatmap=predict[:,:,0]



    fig, axes = plt.subplots(1, 2,figsize=(15,15))

    axes[0].set_axis_off()

    axes[0].imshow(img)

    axes[1].set_axis_off()

    axes[1].imshow(heatmap)

    plt.show()
## NMSとかの関数定義(多くかぶったところを取り除く)



from PIL import Image, ImageDraw



def NMS_all(predicts,category_n,score_thresh,iou_thresh):

    y_c=predicts[...,category_n]+np.arange(pred_out_h).reshape(-1,1)

    x_c=predicts[...,category_n+1]+np.arange(pred_out_w).reshape(1,-1)

    height=predicts[...,category_n+2]*pred_out_h

    width=predicts[...,category_n+3]*pred_out_w



    count=0

    for category in range(category_n):

        predict=predicts[...,category]

        mask=(predict>score_thresh)

        #print("box_num",np.sum(mask))

        if mask.all==False:

            continue

        box_and_score=NMS(predict[mask],y_c[mask],x_c[mask],height[mask],width[mask],iou_thresh)

        box_and_score=np.insert(box_and_score,0,category,axis=1)#category,score,top,left,bottom,right

        if count==0:

            box_and_score_all=box_and_score

        else:

            box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)

        count+=1

        score_sort=np.argsort(box_and_score_all[:,1])[::-1]

        box_and_score_all=box_and_score_all[score_sort]

        #print(box_and_score_all)



    _,unique_idx=np.unique(box_and_score_all[:,2],return_index=True)

    #print(unique_idx)

    return box_and_score_all[sorted(unique_idx)]

  

def NMS(score,y_c,x_c,height,width,iou_thresh,merge_mode=False):

    if merge_mode:

        score=score

        top=y_c

        left=x_c

        bottom=height

        right=width

    else:

        #flatten

        score=score.reshape(-1)

        y_c=y_c.reshape(-1)

        x_c=x_c.reshape(-1)

        height=height.reshape(-1)

        width=width.reshape(-1)

        size=height*width





        top=y_c-height/2

        left=x_c-width/2

        bottom=y_c+height/2

        right=x_c+width/2



        inside_pic=(top>0)*(left>0)*(bottom<pred_out_h)*(right<pred_out_w)

        outside_pic=len(inside_pic)-np.sum(inside_pic)

        #if outside_pic>0:

        #  print("{} boxes are out of picture".format(outside_pic))

        normal_size=(size<(np.mean(size)*10))*(size>(np.mean(size)/10))

        score=score[inside_pic*normal_size]

        top=top[inside_pic*normal_size]

        left=left[inside_pic*normal_size]

        bottom=bottom[inside_pic*normal_size]

        right=right[inside_pic*normal_size]









    #sort  

    score_sort=np.argsort(score)[::-1]

    score=score[score_sort]  

    top=top[score_sort]

    left=left[score_sort]

    bottom=bottom[score_sort]

    right=right[score_sort]



    area=((bottom-top)*(right-left))



    boxes=np.concatenate((score.reshape(-1,1),top.reshape(-1,1),left.reshape(-1,1),bottom.reshape(-1,1),right.reshape(-1,1)),axis=1)



    box_idx=np.arange(len(top))

    alive_box=[]

    while len(box_idx)>0:



        alive_box.append(box_idx[0])



        y1=np.maximum(top[0],top)

        x1=np.maximum(left[0],left)

        y2=np.minimum(bottom[0],bottom)

        x2=np.minimum(right[0],right)



        cross_h=np.maximum(0,y2-y1)

        cross_w=np.maximum(0,x2-x1)

        still_alive=(((cross_h*cross_w)/area[0])<iou_thresh)

        if np.sum(still_alive)==len(box_idx):

            print("error")

            print(np.max((cross_h*cross_w)),area[0])

        top=top[still_alive]

        left=left[still_alive]

        bottom=bottom[still_alive]

        right=right[still_alive]

        area=area[still_alive]

        box_idx=box_idx[still_alive]

    return boxes[alive_box]#score,top,left,bottom,right







def draw_rectangle(box_and_score,img,color):

    number_of_rect=np.minimum(500,len(box_and_score))

  

    for i in reversed(list(range(number_of_rect))):

        top, left, bottom, right = box_and_score[i,:]





        top = np.floor(top + 0.5).astype('int32')

        left = np.floor(left + 0.5).astype('int32')

        bottom = np.floor(bottom + 0.5).astype('int32')

        right = np.floor(right + 0.5).astype('int32')

        #label = '{} {:.2f}'.format(predicted_class, score)

        #print(label)

        #rectangle=np.array([[left,top],[left,bottom],[right,bottom],[right,top]])



        draw = ImageDraw.Draw(img)

        #label_size = draw.textsize(label)

        #print(label_size)



        #if top - label_size[1] >= 0:

        #  text_origin = np.array([left, top - label_size[1]])

        #else:

        #  text_origin = np.array([left, top + 1])



        thickness=4

        if color=="red":

            rect_color=(255, 0, 0)

        elif color=="blue":

            rect_color=(0, 0, 255)

        else:

            rect_color=(0, 0, 0)

      

    

        if i==0:

            thickness=4

        for j in range(2*thickness):#薄いから何重にか描く

            draw.rectangle([left + j, top + j, right - j, bottom - j],

                        outline=rect_color)

            #draw.rectangle(

            #            [tuple(text_origin), tuple(text_origin + label_size)],

            #            fill=(0, 0, 255))

            #draw.text(text_origin, label, fill=(0, 0, 0))



        del draw

        return img



def check_iou_score(true_boxes,detected_boxes,iou_thresh):

    iou_all=[]

    for detected_box in detected_boxes:

        y1=np.maximum(detected_box[0],true_boxes[:,0])

        x1=np.maximum(detected_box[1],true_boxes[:,1])

        y2=np.minimum(detected_box[2],true_boxes[:,2])

        x2=np.minimum(detected_box[3],true_boxes[:,3])



        cross_section=np.maximum(0,y2-y1)*np.maximum(0,x2-x1)

        all_area=(detected_box[2]-detected_box[0])*(detected_box[3]-detected_box[1])+(true_boxes[:,2]-true_boxes[:,0])*(true_boxes[:,3]-true_boxes[:,1])

        iou=np.max(cross_section/(all_area-cross_section))

        #argmax=np.argmax(cross_section/(all_area-cross_section))

    iou_all.append(iou)

    score=2*np.sum(iou_all)/(len(detected_boxes)+len(true_boxes))

    print("score:{}".format(np.round(score,3)))

    return score



def split_and_detect(model,img,height_split_recommended,width_split_recommended,score_thresh=0.3,iou_thresh=0.4):

    width,height=img.size

    pred_in_w,pred_in_h=512,512

    pred_out_w,pred_out_h=128,128

    category_n=1

    maxlap=0.5

    height_split=int(-(-height_split_recommended//1)+1)

    width_split=int(-(-width_split_recommended//1)+1)

    height_lap=(height_split-height_split_recommended)/(height_split-1)

    height_lap=np.minimum(maxlap,height_lap)

    width_lap=(width_split-width_split_recommended)/(width_split-1)

    width_lap=np.minimum(maxlap,width_lap)



    if height>width:

        crop_size=int((height)/(height_split-(height_split-1)*height_lap))#crop_height and width

        if crop_size>=width:

            crop_size=width

            stride=int((crop_size*height_split-height)/(height_split-1))

            top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]

            left_list=[0]

        else:

            stride=int((crop_size*height_split-height)/(height_split-1))

            top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]

            width_split=-(-width//crop_size)

            stride=int((crop_size*width_split-width)/(width_split-1))

            left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]



    else:

        crop_size=int((width)/(width_split-(width_split-1)*width_lap))#crop_height and width

        if crop_size>=height:

            crop_size=height

            stride=int((crop_size*width_split-width)/(width_split-1))

            left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]

            top_list=[0]

        else:

            stride=int((crop_size*width_split-width)/(width_split-1))

            left_list=[i*stride for i in range(width_split-1)]+[width-crop_size]

            height_split=-(-height//crop_size)

            stride=int((crop_size*height_split-height)/(height_split-1))

            top_list=[i*stride for i in range(height_split-1)]+[height-crop_size]

    

    count=0



    for top_offset in top_list:

        for left_offset in left_list:

            img_crop = img.crop((left_offset, top_offset, left_offset+crop_size, top_offset+crop_size))

            predict=model.predict((np.asarray(img_crop.resize((pred_in_w,pred_in_h))).reshape(1,pred_in_h,pred_in_w,3))/255).reshape(pred_out_h,pred_out_w,(category_n+4))

    

            box_and_score=NMS_all(predict,category_n,score_thresh,iou_thresh)#category,score,top,left,bottom,right

            

            #print("after NMS",len(box_and_score))

            if len(box_and_score)==0:

                continue

            #reshape and offset

            box_and_score=box_and_score*[1,1,crop_size/pred_out_h,crop_size/pred_out_w,crop_size/pred_out_h,crop_size/pred_out_w]+np.array([0,0,top_offset,left_offset,top_offset,left_offset])

            

            if count==0:

                box_and_score_all=box_and_score

            else:

                box_and_score_all=np.concatenate((box_and_score_all,box_and_score),axis=0)

            count+=1

    #print("all_box_num:",len(box_and_score_all))

    #print(box_and_score_all[:10,:],np.min(box_and_score_all[:,2:]))

    if count==0:

        box_and_score_all=[]

    else:

        score=box_and_score_all[:,1]

        y_c=(box_and_score_all[:,2]+box_and_score_all[:,4])/2

        x_c=(box_and_score_all[:,3]+box_and_score_all[:,5])/2

        height=-box_and_score_all[:,2]+box_and_score_all[:,4]

        width=-box_and_score_all[:,3]+box_and_score_all[:,5]

        #print(np.min(height),np.min(width))

        box_and_score_all=NMS(box_and_score_all[:,1],box_and_score_all[:,2],box_and_score_all[:,3],box_and_score_all[:,4],box_and_score_all[:,5],iou_thresh=0.5,merge_mode=True)

    return box_and_score_all

from tqdm import tqdm_notebook

from joblib import Parallel,delayed



K.clear_session()

print("loading models...")

model_1=create_model(input_shape=(512,512,3),size_detection_mode=True)

model_1.load_weights('../input/models/centernet_step1.h5')

#model_1.load_weights('./input/models/centernet_step1.h5')



model_2=create_model(input_shape=(512,512,3),size_detection_mode=False)

model_2.load_weights('../input/models/centernet_step2.h5')

#model_1.load_weights('./models/centernet_step2.h5')



def pipeline(i):

    # model1: determine how to split image

    img = np.asarray(Image.open(id_test[i]).resize((512,512)).convert('RGB'))

    predicted_size=model_1.predict(img.reshape(1,512,512,3)/255)

    detect_num_h=aspect_ratio_pic_all_test[i]*np.exp(-predicted_size/2)

    detect_num_w=detect_num_h/aspect_ratio_pic_all_test[i]

    h_split_recommend=np.maximum(1,detect_num_h/base_detect_num_h)

    w_split_recommend=np.maximum(1,detect_num_w/base_detect_num_w)



    # model2: detection

    img=Image.open(id_test[i]).convert("RGB")

    box_and_score_all=split_and_detect(model_2,img,h_split_recommend,w_split_recommend,score_thresh=0.3,iou_thresh=0.4)#output:score,top,left,bottom,right

    lists=[]

    if (len(box_and_score_all)>0):

        for box in box_and_score_all[:,1:]:

            top,left,bottom,right=box

            lists.append([left,top,right-left,bottom-top])

    df=pd.DataFrame(lists,columns=["X","Y","width","height"])

    df["image_id"]=os.path.basename(id_test[i]).split(".")[0]

    df["label"]=""

    df=df[["image_id","label","X","Y","width","height"]]

    return df

print("predicts...")

#I'm sorry. Not nice coding. Time consuming.

bboxes_df=pd.concat([pipeline(i) for i in tqdm_notebook(range(len(id_test)))])
bboxes_df.to_pickle("./test_centernet_p3.pkl")
rcParams['figure.figsize'] = 10,10

image_write(1,bboxes_df,folder="test_images",label_show=False)
import gc

K.clear_session()

del model

gc.collect()
import matplotlib

import pandas as pd

import numpy as np

from PIL import Image

from pylab import rcParams

import os

import json

import matplotlib

import matplotlib.pyplot as plt

import glob



%matplotlib inline



import torch

import os

import pandas as pd

import pickle

import torchvision.transforms as transforms

from torch.utils.data import Dataset

from PIL import Image

from tqdm import tqdm_notebook

from cnn_finetune import make_model

import torch.nn as nn

from torch.autograd import Variable

import torch.nn as nn

import torch.optim as optim



ut=pd.read_csv("../input/kuzushiji-recognition/unicode_translation.csv")

ut_dict=ut.set_index("Unicode")["char"].to_dict()

train=pd.read_csv("../input/kuzushiji-recognition/train.csv")

train_labels["id"]=np.arange(len(train_labels))
# 同じラベルが5以下のものは5個に、400以上のものは400にする

id_list=[]

for l,sdf in train_labels.groupby("label"):

    image_names=sdf["id"].values

    if len(sdf)<5:

        image_names=list(image_names)+list(np.random.choice(sdf["id"].values, 5-len(sdf)))

    elif len(sdf)>400:

        image_names=np.random.choice(sdf["id"].values, 400,replace=False)

    id_list+=list(image_names)

train_labels_m=train_labels.set_index("id").loc[id_list].reset_index()
# trainとvalに分割

from sklearn.model_selection import train_test_split

train_labels_train, train_labels_val, _, _ = train_test_split(train_labels_m,\

                                                    train_labels_m["label"],\

                                                    test_size=0.2,\

                                                    random_state=100,\

                                                    stratify=train_labels_m["label"])

# labelを数値に変換(一応保存)

label_id={label:i for i,label in enumerate(train_labels["label"].unique())}

with open("./label_id.pkl","wb") as f:

    pickle.dump(label_id,f)
with open("./label_id.pkl","rb") as f:

    label_id=pickle.load(f)

label_id_r={v:k for k,v in label_id.items()}
resize = (256, 256)  # 入力画像サイズ

train_dir="./train_images"

trans= [transforms.Resize(resize),

#            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),

            transforms.Grayscale(num_output_channels=3),

            transforms.RandomAffine(5,translate=(0.1,0.1),fillcolor="white"),

            transforms.RandomCrop((224,224),fill="white"),

#            transforms.RandomRotation(degrees=5,fill="white"),

            transforms.Resize(resize),

            transforms.ToTensor(),

            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]





class MyDataSet(Dataset):

    def __init__(self,img_dir,train_labels):

        self.train_labels = train_labels

        self.transform = transforms.Compose(trans)

        self.img_dir=img_dir

        self.images = list(self.train_labels["id"].unique())

        self.labels = list(self.train_labels["label"].unique())

      

    def __len__(self):

        return len(self.train_labels)

    

    def image_open(self,t):

        image = Image.open(os.path.join(self.img_dir, t+".jpg"))

        return image.convert('RGB')



    def __getitem__(self, idx):

        image_id,X,Y,width,height,label = self.train_labels[["image_id","X","Y","width","height","label"]].iloc[idx]

        img = Image.open( os.path.join(self.img_dir, image_id+".jpg") )

        if width < height:

            img_crop = img.crop((X+(width-height)//2, Y, X+(width+height)//2, Y+height))

        else:

            img_crop = img.crop((X, Y+(height-width)/2, X+width, Y+(height+width)/2))

        

        return self.transform(img_crop),label_id[label]



kwargs = {'num_workers': 1, 'pin_memory': True} 

train_set = MyDataSet(train_dir,train_labels_train)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True)

val_set = MyDataSet(train_dir,train_labels_val)

val_loader = torch.utils.data.DataLoader(val_set, batch_size=5, shuffle=True)

dataloaders_dict={"train":train_loader,"val":val_loader}
# transform可視化用関数

rcParams['figure.figsize'] = 4,4

def tran_picture(idx):

    print("words:",ut_dict[train_labels.iloc[idx]["label"]])

    image_id,X,Y,width,height = train_labels[["image_id","X","Y","width","height"]].iloc[idx]

    img = Image.open( os.path.join(train_dir, image_id+".jpg") )

    if width < height:

            img_crop = img.crop((X+(width-height)//2, Y, X+(width+height)//2, Y+height))

    else:

            img_crop = img.crop((X, Y+(height-width)/2, X+width, Y+(height+width)/2))

    p=img_crop

    plt.imshow(p)

    plt.show()

    img_transformed=transforms.Compose(trans)(p)

    img_transformed = img_transformed.numpy().transpose((1, 2, 0))

    img_transformed = np.clip(img_transformed, 0, 1)

    plt.imshow(img_transformed)

    plt.show()
rcParams['figure.figsize']=[4,4]

for i in range(10):

    print(tran_picture(i))

resize = (256, 256)  # 入力画像サイズ



class Identity(nn.Module):

    def __init__(self):

        super(Identity, self).__init__()        



    def forward(self, x):

        return x





def make_pnas():

# 実際はpnasnet5leargeにしましたが、遅くなるため今回はresnetで

#    model = make_model('pnasnet5large', pretrained=True, input_size=resize,num_classes=4212)

    model = make_model('resnet101', pretrained=False, input_size=resize,num_classes=4212)

    return model

model = make_pnas()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

model.cuda()
# epochのループ

num_epochs=0

start_num=0

model_path=0

#model.load_state_dict(torch.load('./models/model7/model-epoch-3.pth'))

if not os.path.exists("./models/model{}".format(model_path)):

    os.makedirs("./models/model{}".format(model_path))



net, dataloaders_dict, criterion, optimizer=model, dataloaders_dict, criterion, optimizer

for epoch in range(num_epochs):

    print('Epoch {}/{}'.format(epoch+1, num_epochs))

    print('-------------')



    # epochごとの学習と検証のループ

    for phase in ['train', 'val']:

        if phase == 'train':

            net.train()  # モデルを訓練モードに

        else:

            net.eval()   # モデルを検証モードに



        epoch_loss = 0.0  # epochの損失和

        epoch_corrects = 0  # epochの正解数



        # データローダーからミニバッチを取り出すループ

        for inputs, labels in tqdm_notebook(dataloaders_dict[phase]):

            inputs,labels = inputs.cuda(),labels.cuda()

            # optimizerを初期化

            optimizer.zero_grad()



            # 順伝搬（forward）計算

            with torch.set_grad_enabled(phase == 'train'):

                outputs = net(inputs)

                outputs=outputs

                loss = criterion(outputs, labels)  # 損失を計算

                _, preds = torch.max(outputs, 1)  # ラベルを予測





                # 訓練時はバックプロパゲーション

                if phase == 'train':

                    loss.backward()

                    optimizer.step()



                # イタレーション結果の計算

                # lossの合計を更新

                epoch_loss += loss.item() * inputs.size(0)  

                # 正解数の合計を更新

                epoch_corrects += torch.sum(preds == labels.data)



        # epochごとのlossと正解率を表示

        epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)

        epoch_acc = epoch_corrects.double(

        ) / len(dataloaders_dict[phase].dataset)



        print('{} Loss: {:.4f} Acc: {:.4f}'.format(

            phase, epoch_loss, epoch_acc))

            

    torch.save(model.state_dict(), './models/model{}/model-epoch-{}.pth'.format(model_path,epoch))
resize=(256,256)

test_dir="./test_images"

test_bboxes_df = pd.read_pickle("./test_centernet_p3.pkl")

model.load_state_dict(torch.load('../input/models/resnet-trained.pth'))

model.cuda()



trans= [transforms.Resize(resize),

#            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),

            transforms.Grayscale(num_output_channels=3),

            transforms.Resize(resize),

            transforms.ToTensor(),

            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

class MyDataSet_test(Dataset):

    def __init__(self,img_dir,train_labels,groupbysample_n=False):

        if groupbysample_n:

            self.train_labels = train_labels.groupby("label").apply(lambda x: x.sample(n=groupbysample_n,replace=True))

        else:

            self.train_labels = train_labels

        self.transform = transforms.Compose(trans)

        self.img_dir=img_dir

        

    def __len__(self):

        return len(self.train_labels)

    

    def __getitem__(self, idx):

        image_id,X,Y,width,height = self.train_labels[["image_id","X","Y","width","height"]].iloc[idx]

        img = Image.open( os.path.join(self.img_dir, image_id+".jpg") )

        if width < height:

            img_crop = img.crop((X+(width-height)//2, Y, X+(width+height)//2, Y+height))

        else:

            img_crop = img.crop((X, Y+(height-width)/2, X+width, Y+(height+width)/2))

        

        return self.transform(img_crop), image_id,X,Y,width,height

    

test_set = MyDataSet_test(test_dir, test_bboxes_df)

test_loader = torch.utils.data.DataLoader(test_set,batch_size=3, shuffle=False)

lists=[]

for i,(x,image_ids,Xs,Ys,widths,heights) in tqdm_notebook(enumerate(test_loader),total=len(test_loader)):

    a=model(x.cuda())

    a=torch.max(a,1)

    df=pd.DataFrame(a[1].cpu().detach().numpy(),columns=["labels_id"])

    df["label"]=df["labels_id"].map(label_id_r)

    df["image_id"]=image_ids

    df["X"]=Xs

    df["Y"]=Ys

    df["width"]=widths

    df["height"]=heights

    lists.append(df[["image_id","X","Y","width","height","label"]])

# 予測時間長いので、今回は1000でストップ

    if i>=1000:

        break

test_labels_p=pd.concat(lists)

test_labels_p.to_pickle("./test_labels_p4.pkl")
rcParams['figure.figsize']=[10,10]

image_write(0,test_labels_p,folder="test_images")
## 提出用データ

import glob

data=[]

for image,sdf in test_labels_p[["image_id","X","Y","width","height","label"]].groupby("image_id"):

    labels=" ".join(["{} {} {}".format(l,int(X+1/2*w),int(Y+1/2*h)) for image,X,Y,w,h,l in sdf.values])

    data.append([image,labels])

df=pd.DataFrame(data,columns=["image_id","labels"])

test_imgs=glob.glob("./test_images/*")

test_img_df=pd.DataFrame([test_img.split("/")[-1].split(".")[0] for test_img in test_imgs],columns=["image_id"])

df=pd.merge(test_img_df,df,how="left",on="image_id").fillna("")

df.to_csv("./prediction.csv",index=False)
!rm -R train_images

!rm -R test_images

!rm -R models