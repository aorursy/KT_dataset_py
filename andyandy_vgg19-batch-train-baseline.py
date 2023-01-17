import numpy as np

from PIL import Image

import pandas as pd



import os

#print(os.listdir("../input/vgg19"))



train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv") #.sample(200)

test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv") #.sample(200)

submit = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv") #.sample(200)

diagnosis_encoded = pd.get_dummies(train.diagnosis)

from keras.applications.vgg19 import VGG19

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array

from keras.applications.vgg19 import preprocess_input

from keras.models import Model, Sequential

from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D

from keras.layers import MaxPooling2D, Flatten, Dense



vgg19 = VGG19(weights='../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',include_top=False,input_shape=(224,224,3))

for l in vgg19.layers: 

    if l is not None: l.trainable = False 

        

x = vgg19.output

x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = Dense(256, activation='relu')(x)

predictions = Dense(5, activation='softmax')(x)



model = Model(inputs=vgg19.input, outputs=predictions)        

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
x_train_list=[]

y_train_list=[]

# -------------------------------------------

for index, row in train.iterrows():

    my_pic_name = row.id_code

    im = Image.open("../input/aptos2019-blindness-detection/train_images/"+my_pic_name+".png")

    im_224 = im.resize((224,224), Image.ANTIALIAS)

    

    image = img_to_array(im_224)

    # image = image.reshape((image.shape[0], image.shape[1], image.shape[2]))    

    #image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)  

    

    x_train_list.append(image)

    y_train_list.append(diagnosis_encoded.loc[index])

    

    #print(index)

    #print(diagnosis_encoded.loc[index])

    #print(np.array(x_train_list).shape)

    #print(np.array(y_train_list).shape)



    x_train_raw = np.array(x_train_list, np.float32) / 255.    

    y_train_raw = np.array(y_train_list, np.uint8)

    

    if len(x_train_list)%200==0:   

        model.train_on_batch(x_train_raw, y_train_raw)

        x_train_list=[]

        y_train_list=[]

        print('train on batch ...')

# -------------------------------------------
y_class = []

for index, row in test.iterrows():

    my_pic_name = row.id_code

    im = Image.open("../input/aptos2019-blindness-detection/test_images/"+my_pic_name+".png")

    im_224 = im.resize((224,224), Image.ANTIALIAS)

    

    image = img_to_array(im_224)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))    

    #image = np.expand_dims(image, axis=0)

    image = preprocess_input(image)  

    

    y_class.append(np.argmax(model.predict(image), axis=1)[0])

# -------------------------------------------
submit.diagnosis = y_class

submit.to_csv('submission.csv',index=False)