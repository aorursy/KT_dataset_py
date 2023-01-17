# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import numpy as np

import os

import cv2

import pandas as pd

import joblib

from pathlib import Path

from keras.applications.vgg16 import preprocess_input

from keras.applications import  vgg16

from keras.models import  Model



from keras.applications import vgg16

from keras.preprocessing import image

from keras.layers import Dense,Flatten,Dropout,InputLayer

from keras.models import Sequential



from keras import optimizers

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# from keras.preprocessing import image

# import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
img = image.load_img("../input/cell_images/cell_images/Uninfected/C214ThinF_IMG_20151106_131748_cell_148.png")

img2 = image.load_img("../input/cell_images/cell_images/Parasitized/C140P101ThinF_IMG_20151005_211735_cell_159.png")
def load_images_from_folder(folder,lent):

    

    

    count = 0

    images = []

    

    

    for filename in os.listdir(folder):



#        img = cv2.imread(os.path.join(folder,filename))

        img = image.load_img(os.path.join(folder,filename),target_size=(224,224))

        img = image.img_to_array(img)

        img = preprocess_input(img)

        

        

        if img is not None:

            images.append(img)

        

        count = count + 1

        

        if count == lent:

            break

            

    return images



def array_to_df(arr_as_list,label_name):

    

    temp_arr = np.array(arr_as_list)

    temp_arr = np.reshape(temp_arr,(temp_arr.shape[0],224*224*3 ) )

    

    

        

    temp_label =[]

    

    for i in range(0,temp_arr.shape[0]):

        temp_label.append(label_name)

    

    temp_label  = np.asarray(temp_label)

    

    image_df = pd.DataFrame(temp_arr)

    label_df = pd.DataFrame(temp_label)    

        

    total_df = pd.concat([image_df,label_df],axis=1)

    

    return total_df



Uninfected = load_images_from_folder('../input/cell_images/cell_images/Uninfected',500)

Parasitized = load_images_from_folder('../input/cell_images/cell_images/Parasitized',500)



Uninfected_df = array_to_df(Uninfected,'Uninf')

Parasitized_df = array_to_df(Parasitized,'PrasiteInside')



total_df = pd.concat([Uninfected_df,Parasitized_df],axis=0)

total_array = np.array(total_df)





x = total_array[:,0:-1]

y = total_array[:,-1]



from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

lb.fit(y)

y = lb.transform(y)





print(x.shape)

print(y.shape)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)



print('x_train : ', x_train.shape)

print('x_test : ', x_test.shape)

print('y_train : ', y_train.shape)

print('y_test : ', y_test.shape)
x_train = x_train/225

x_test = x_test/225



vgg = vgg16.VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3)) 

 

output = vgg.layers[-1].output 

output = Flatten()(output) 

vgg_model = Model(vgg.input, output) 

vgg_model.trainable = False 

 

for layer in vgg_model.layers: 

    layer.trainable = False 

 

vgg_model.summary() 





def get_bottleneck_features(model, input_imgs): 

    features = model.predict(input_imgs, verbose=0) 

    return features 



train_features_vgg = get_bottleneck_features(vgg_model, np.reshape(x_train,(750,224,224,3))) 

test_features_vgg = get_bottleneck_features(vgg_model, np.reshape(x_test,(250,224,224,3))) 



input_shape = vgg_model.output_shape[1] 

model = Sequential() 

model.add(InputLayer(input_shape=(input_shape,))) 

model.add(Dense(512, activation='relu', input_dim=input_shape))

model.add(Dropout(0.3)) 

model.add(Dense(512, activation='relu')) 

model.add(Dropout(0.5)) 

model.add(Dense(1, activation='sigmoid')) 

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy']) 



model.summary()
history = model.fit(train_features_vgg,y_train,epochs=30,validation_data=(test_features_vgg,y_test),steps_per_epoch=10,validation_steps=50)
def single_predict(vgg_model,model,i):

    features = get_bottleneck_features(vgg_model, np.reshape(image.img_to_array(i),(1,224,224,3)))

    pred = model.predict(features)

    return pred

    
single_ans = single_predict(vgg_model,model,image.load_img('../input/cell_images/cell_images/Parasitized/C140P101ThinF_IMG_20151005_211735_cell_159.png',target_size=(224,224,3)))
lb.classes_[int(single_ans[0][0])]

#output


    