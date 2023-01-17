import os
import gc
print(os.listdir("../input"))
import numpy as np 
import pandas as pd
import time
train = pd.read_csv('../input/train_ship_segmentations_v2.csv')
train.head()
train['exist_ship'] = train['EncodedPixels'].fillna(0)
train.loc[train['exist_ship']!=0,'exist_ship']=1
del train['EncodedPixels']
print(len(train['ImageId']))
print(train['ImageId'].value_counts().shape[0])
train_gp = train.groupby('ImageId').sum().reset_index()
train_gp.loc[train_gp['exist_ship']>0,'exist_ship']=1
print(train_gp['exist_ship'].value_counts())
train_gp= train_gp.sort_values(by='exist_ship')
train_gp = train_gp.drop(train_gp.index[0:100000])
print(train_gp['exist_ship'].value_counts())
train_sample = train_gp.sample(5000)
print(train_sample['exist_ship'].value_counts())
print (train_sample.shape)
Train_path = '../input/train_v2/'
Test_path = '../input/test_v2/'
%%time
training_img_data = []
target_data = []
from PIL import Image
data = np.empty((len(train_sample['ImageId']),256, 256,3), dtype=np.uint8)
data_target = np.empty((len(train_sample['ImageId'])), dtype=np.uint8)
image_name_list = os.listdir(Train_path)
index = 0
for image_name in image_name_list:
    if image_name in list(train_sample['ImageId']):
        imageA = Image.open(Train_path+image_name).resize((256,256)).convert('RGB')
        data[index]=imageA
        data_target[index]=train_sample[train_gp['ImageId'].str.contains(image_name)]['exist_ship'].iloc[0]
        index+=1
        
print(data.shape)
print(data_target.shape)
from sklearn.preprocessing import OneHotEncoder
targets =data_target.reshape(len(data_target),-1)
enc = OneHotEncoder()
enc.fit(targets)
targets = enc.transform(targets).toarray()
print(targets.shape)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(data,targets, test_size = 0.2)
x_train.shape, x_val.shape, y_train.shape, y_val.shape
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    zca_whitening = False,
    rotation_range = 90,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    brightness_range = [0.5, 1.5],
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True
    
)
#from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
#from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
from keras.applications.resnet50 import ResNet50 as ResModel
#from keras.applications.vgg16 import VGG16 as VGG16Model
img_width, img_height = 256, 256
model = ResModel(weights = 'imagenet', include_top=False, input_shape = (img_width, img_height, 3))
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential, Model 
for layer in model.layers:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)
from keras import optimizers
epochs = 10
lrate = 0.001
decay = lrate/epochs
#adam = optimizers.Adam(lr=lrate,beta_1=0.9, beta_2=0.999, decay=decay)
sgd = optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model_final.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model_final.summary()
model_final.fit_generator(img_gen.flow(x_train, y_train, batch_size = 16),steps_per_epoch = len(x_train)/16, validation_data = (x_val,y_val), epochs = epochs )
model_final.save('ResNet_transfer_ship.h5')
gc.collect()
train_predict_sample = train_gp.sample(2000)
print(train_predict_sample['exist_ship'].value_counts())
%%time
from PIL import Image
data_predict = np.empty((len(train_predict_sample['ImageId']),256, 256,3), dtype=np.uint8)
data_target_predict = np.empty((len(train_predict_sample['ImageId'])), dtype=np.uint8)
image_name_list = os.listdir(Train_path)
index = 0
for image_name in image_name_list:
    if image_name in list(train_predict_sample['ImageId']):
        imageA = Image.open(Train_path+image_name).resize((256,256)).convert('RGB')
        data_predict[index]=imageA
        data_target_predict[index]=train_predict_sample[train_gp['ImageId'].str.contains(image_name)]['exist_ship'].iloc[0]
        index+=1
        
print(data_predict.shape)
print(data_target_predict.shape)
from sklearn.preprocessing import OneHotEncoder
targets_predict =data_target_predict.reshape(len(data_target_predict),-1)
enc = OneHotEncoder()
enc.fit(targets_predict)
targets_predict = enc.transform(targets_predict).toarray()
print(targets_predict.shape)
predict_ship = model_final.evaluate(data_predict,targets_predict)
print ('Accuracy of random data = '+ str(round(predict_ship[1]*100)) + "%")
image_test_name_list = os.listdir(Test_path)
data_test = np.empty((len(image_test_name_list),256, 256,3), dtype=np.uint8)
test_name = []
index = 0
for image_name in image_test_name_list:
    imageA = Image.open(Test_path+image_name).resize((256,256)).convert('RGB')
    test_name.append(image_name)
    data_test[index]=imageA
    index+=1
print (data_test.shape)
result = model_final.predict(data_test)
result_list={
    "ImageId": test_name,
    "Have_ship":np.argmax(result,axis=1)
}
result_pd = pd.DataFrame(result_list)
result_pd.to_csv('Have_ship_or_not.csv',index = False)