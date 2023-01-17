# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
os.chdir("../input")
os.listdir()

# Any results you write to the current directory are saved as output.
import keras
from keras.layers import *
from keras.optimizers import Adam,Adadelta
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
train_data=ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input,
                          horizontal_flip=True,
                          rotation_range=40,
                          width_shift_range=0.2,
                          height_shift_range=0.2,
                          shear_range=0.2,
                          zoom_range=0.2,
                          fill_mode='nearest',
                          validation_split=0.3)

train=train_data.flow_from_directory('hackexpo2018/train/train/',target_size=(224,224),batch_size=30,
                                     color_mode='rgb',subset='training',class_mode='categorical')

validate=train_data.flow_from_directory('hackexpo2018/train/train/',target_size=(224,224),batch_size=30
                                        ,color_mode='rgb',subset='validation',class_mode='categorical')
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
input_tensor=Input((224,224,3))
base_model=ResNet50(input_tensor=input_tensor,pooling='max',weights='resnet50/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
model=Sequential()
model.add(base_model)

#model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation='softmax'))

model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = Adadelta(), metrics = ['accuracy'])
model_path = "my-model_{epoch:03d}.h5"

checkpoint = ModelCheckpoint(filepath= model_path, monitor = 'val_acc', save_best_only = True, save_weights_only = True, verbose = 1)

model.fit_generator(train,steps_per_epoch=int(2520/30),epochs=20,validation_data=validate,validation_steps=int(1080 /30),callbacks=[checkpoint])
score=model.evaluate_generator(validate,steps=50,verbose=0)

print ('Accuracy :', score[1])
print ('loss :', score[0])
from glob import glob as gb

models=sorted(gb('*h5'))[-1]

model.load_weights(models)
print (models)
from PIL import Image
def test_data_preparation(dir):
    dir='hackexpo2018/test/'+dir
    
    img_arr=[]
    img_id=gb(dir+'/*')

    for i in img_id:
        img_arr.append(np.asarray(Image.open(i).resize((224,224))).flatten())
        pass
    img_arr=np.array(img_arr)/255
    return [img_arr,img_id]
img_arr,imag_id=test_data_preparation('test')
ls
predictions=model.predict(img_arr.reshape((-1,224,224,3)))
predictions
import numpy as np
predictions=np.argmax(predictions,1)
import pandas as pd

submission=pd.DataFrame({'ImageID':imag_id,'Category':predictions})
submission.ImageID=submission.ImageID.apply(lambda x:x[23:])
submission.to_csv('sadiq.csv',index=False)