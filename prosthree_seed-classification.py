import tensorflow as tf
import os
import cv2
import random
import numpy as np
import pandas as pd

from tqdm import trange
from tqdm import tqdm
from tensorflow.keras import Model
from tensorflow.keras.layers import multiply,Reshape,DepthwiseConv2D,Concatenate,Dense, Flatten, Conv2D,Input,GlobalAveragePooling2D,add,Add,MaxPooling2D,Dropout,ZeroPadding2D
from keras.layers.core import Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.metrics import CategoricalAccuracy
from keras import backend as K
from time import sleep

train_dir = "../input/plant-seedlings-classification/train"
test_dir = "../input/plant-seedlings-classification/test"

class_list = os.listdir(train_dir)

image_shape = (224,224,3)

batch_size = 4
epoch = 20
def load_dataset(data_dir,class_list=None):
    data = []
    if class_list == None:
        for img in os.listdir(os.path.join(data_dir)):
            data.append(os.path.join(data_dir,img))
    else:
        for i,class_name in enumerate(class_list):
            for img in os.listdir(os.path.join(data_dir,class_name)):
                data.append([os.path.join(data_dir,class_name,img),i])

    return data

def validation_split(data,ratio = 0.2):
    random.shuffle(data)
    train_data = data[int(ratio*len(data))+1:]
    val_data = data[:int(ratio*len(data))]

    
    return train_data,val_data

def load_img(img_path,resize=(224,224),rescale=1.0):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(resize))
    img = img*rescale
    return img

def load_batch(data,batch_size = 1,shuffle=False):
    result = []

    if shuffle:
        random.shuffle(data)

    for i in range(len(data)//batch_size):
        batch_data = []
        for j in range(batch_size):
            batch_data.append(data[int((i*batch_size + j)%len(data))])
        result.append(batch_data)

    return result

def load_batch_data(data,img_resize=(224,224),rescale=1.0):
    x = []
    y = []

    for img_data in data:
        x.append(load_img(img_data[0],resize=img_resize,rescale=1/255))
        y.append(img_data[-1])

    return np.array(x),np.array(y)
def swish_activation(x):
    return x*K.sigmoid(x)
def basic_block(model, kernal_size, filters, output_filters, strides,expand_ratio = 1,squeeze_ratio=0.25):
    prev_model = model
    
    model = Conv2D(filters = filters*expand_ratio, kernel_size = 1, strides = strides, padding = "same")(prev_model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = Activation(swish_activation)(model)
    
    model = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)
    model = Activation(swish_activation)(model)
    
    se = GlobalAveragePooling2D()(model)
    se = Reshape((1,1,filters*expand_ratio))(se)
    
    squeezed_filters = max(1, int(filters * squeeze_ratio))
    se = Conv2D(filters = squeezed_filters, kernel_size = 1, strides = strides, padding = "same")(se)
    se = Activation(swish_activation)(se)
    
    se = Conv2D(filters = filters*expand_ratio, kernel_size = 1, strides = strides, padding = "same")(se)
    se = Activation('sigmoid')(se)
    
    model = multiply([model,se])
    
    model = Conv2D(filters = output_filters, kernel_size = 1, strides = strides, padding = "same")(model)
    model = BatchNormalization(momentum = 0.5)(model)

    return model

class Model_Generator(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):

        model_input = Input(shape = self.input_shape)
#         224
        model = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = "same")(model_input)
    
#         112
        model = Conv2D(filters = 32, kernel_size = 1, strides = 1,padding='valid')(model)    
        model = basic_block(model,(3,3),filters = 32,output_filters=16,strides=1,expand_ratio=1)
        model = basic_block(model,(3,3),filters = 16,output_filters=24,strides=1,expand_ratio=6)
        model = basic_block(model,(3,3),filters = 16,output_filters=24,strides=1,expand_ratio=6)
        
#         56
        model = Conv2D(filters = 24, kernel_size = 1, strides = 1,padding='valid')(model)
        model = basic_block(model,(5,5),filters = 24,output_filters=40,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 24,output_filters=40,strides=1,expand_ratio=6)
        
#         28
        model = Conv2D(filters = 40, kernel_size = 1, strides = 1,padding='valid')(model)
        model = basic_block(model,(3,3),filters = 40,output_filters=80,strides=1,expand_ratio=6)
        model = basic_block(model,(3,3),filters = 40,output_filters=80,strides=1,expand_ratio=6)
        model = basic_block(model,(3,3),filters = 40,output_filters=80,strides=1,expand_ratio=6)
        
#         14
        model = Conv2D(filters = 80, kernel_size = 1, strides = 1,padding='valid')(model)
        model = basic_block(model,(5,5),filters = 80,output_filters=112,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 80,output_filters=112,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 80,output_filters=112,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 80,output_filters=192,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 80,output_filters=192,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 80,output_filters=192,strides=1,expand_ratio=6)
        model = basic_block(model,(5,5),filters = 80,output_filters=192,strides=1,expand_ratio=6)
        
#         7
        model = Conv2D(filters = 192, kernel_size = 1, strides = 1,padding='valid')(model)
        model = basic_block(model,(3,3),filters = 192,output_filters=320,strides=1,expand_ratio=6)
        
        model = Conv2D(filters = 1280, kernel_size = 1, strides = 1,padding='same')(model)
        model = GlobalAveragePooling2D()(model)
#         model = Dense(1024)(model)
        model = Dense(len(class_list))(model)
        
        model_output = Activation('softmax')(model)

        model = Model(inputs = model_input, outputs = model_output)

        return model
data = load_dataset(train_dir,class_list)
train_data,val_data = validation_split(data)

train_data = load_batch(train_data,batch_size,True)
val_data = load_batch(val_data,batch_size,True)
model = Model_Generator(image_shape).get_model()
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam')
model.summary()
for e in range(epoch*8):
    step_per_e = trange(len(train_data)//8)
    for j in step_per_e:
        step_per_e.set_description("Epoch {}".format(e))
        x,y = load_batch_data(train_data[int(np.random.uniform(low=0, high=len(train_data)))])
        train_losses = model.train_on_batch(x, y)
    
#   Validation Process
    accuracy_acc = CategoricalAccuracy()
    for val in val_data:
        x,y = load_batch_data(val)
        
        y_pred = model.predict(x)
        y_gt = to_categorical(y, num_classes=len(class_list), dtype="float32")
        
        accuracy_acc.update_state(y_pred,y_gt)
    val_losses = model.train_on_batch(x, y)
    print("Train loss : {}, Validation loss : {}, Accuracy : {}".format(train_losses,val_losses,accuracy_acc.result().numpy()))
model.save("trained_model.h5")
test_data = val_data

for i in trange(len(test_data)):
    x,y = load_batch_data(val)

    y_pred = model.predict(x)
    y_gt = to_categorical(y, num_classes=len(class_list), dtype="float32")

    accuracy_acc.update_state(y_pred,y_gt)
print("Model Accuracy : {}".format(accuracy_acc.result().numpy()))
    
test_data = load_dataset(test_dir)
test_data = load_batch(test_data)
res = []
for test in tqdm(test_data):
    img = load_img(test[0],(224,224),1/255)
    y_pred = model.predict(np.expand_dims(img,axis=0))
    class_pred = class_list[int(np.argmax(y_pred))]
    res.append([os.path.basename(test[0]),class_pred])
df = pd.DataFrame(res, columns = ['file', 'species'])
df.to_csv("submission_4.csv",index=False)
!kaggle competitions submit -c plant-seedlings-classification -f submission.csv -m "Message"


