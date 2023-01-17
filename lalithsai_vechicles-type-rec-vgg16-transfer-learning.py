# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/vehicles/vehicles/vehicles"))

print(os.listdir("../input/testautomobileimages/test images/test images"))





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



# Any results you write to the current directory are saved as output.


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



images_bus = load_images_from_folder('../input/vehicles/vehicles/vehicles/bus',200)

images_bikes = load_images_from_folder('../input/vehicles/vehicles/vehicles/bikes',200)

images_cars = load_images_from_folder('../input/vehicles/vehicles/vehicles/cars',200)



print(type(images_bus))



bus_df = array_to_df(images_bus,'bus')

bikes_df = array_to_df(images_bikes,'bikes')

cars_df = array_to_df(images_cars,'cars')







total_df = pd.concat([bus_df,bikes_df,cars_df],axis=0)

#print(total_df)



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
#Feature Extraction way of transfer learning

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



train_features_vgg = get_bottleneck_features(vgg_model, np.reshape(x_train,(450,224,224,3))) 

test_features_vgg = get_bottleneck_features(vgg_model, np.reshape(x_test,(150,224,224,3))) 



input_shape = vgg_model.output_shape[1] 

model = Sequential() 

model.add(InputLayer(input_shape=(input_shape,))) 

model.add(Dense(512, activation='relu', input_dim=input_shape))

model.add(Dropout(0.3)) 

model.add(Dense(512, activation='relu')) 

model.add(Dropout(0.5)) 

model.add(Dense(3, activation='sigmoid')) 

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['accuracy']) 



model.summary()
history2 = model.fit(train_features_vgg,y_train,epochs=30,validation_data=(test_features_vgg,y_test))
bike_test = image.load_img('../input/testautomobileimages/test images/test images/bike.jpg',target_size=(224,224))

bus_test = image.load_img('../input/testautomobileimages/test images/test images/bus.jpg',target_size=(224,224))

car_test = image.load_img('../input/testautomobileimages/test images/test images/car.jpg',target_size=(224,224))
plt.imshow(car_test)
plt.imshow(bus_test)
plt.imshow(bike_test)
bike_arr = image.img_to_array(bike_test)

bike_preprocessed = vgg16.preprocess_input(bike_arr)



bus_arr = image.img_to_array(bus_test)

bus_preprocessed = vgg16.preprocess_input(bus_arr)



car_arr = image.img_to_array(car_test)

car_preprocessed = vgg16.preprocess_input(car_arr)

bike_test_features = get_bottleneck_features(vgg_model, np.reshape(bike_preprocessed,(1,224,224,3)))

bus_test_features = get_bottleneck_features(vgg_model, np.reshape(bus_preprocessed,(1,224,224,3)))

car_test_features = get_bottleneck_features(vgg_model, np.reshape(car_preprocessed,(1,224,224,3)))
model.predict(bike_test_features)
model.predict_classes(bike_test_features)
lb.classes_
lb.classes_[model.predict_classes(bike_test_features)[0]]
class_list = ['bike','bus','car']
print(class_list[model.predict_classes(bike_test_features)[0]])

print(class_list[model.predict_classes(bus_test_features)[0]])

print(class_list[model.predict_classes(car_test_features)[0]])
model.predict_proba(car_test_features)