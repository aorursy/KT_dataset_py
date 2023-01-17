# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from glob import glob

import os

import cv2



from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss

from keras.wrappers.scikit_learn import KerasClassifier

from keras import backend as K

from keras.models import Sequential

from keras.layers.core import Activation, Dense, Dropout, Flatten

from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D

from keras import optimizers

from multiprocessing import Pool, cpu_count

import PIL

from PIL import ImageFilter, ImageStat, Image, ImageDraw



from multiprocessing import Pool, cpu_count





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
def get_im_cv2(path):

    img = cv2.imread(path)    

    resized = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)

    return [path, resized]



def normalize_image_features(paths):

    imf_d = {}

    p = Pool(cpu_count())

    ret = p.map(get_im_cv2, paths)

    for i in range(len(ret)):

        imf_d[ret[i][0]] = ret[i][1]

    ret = []

    fdata = [imf_d[f] for f in paths]

    fdata = np.array(fdata, dtype=np.uint8)

    fdata = fdata.transpose((0, 3, 1, 2))

    fdata = fdata.astype('float32')

    fdata = fdata / 255

    return fdata

    





K.set_image_dim_ordering('tf')



img_width, img_height = 32, 32 



nb_train_samples = 1481



epochs = 50



batch_size = 1



ImageFile.LOAD_TRUNCATED_IMAGES = True



CAT_COLUMN = 'type'



if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)

    

    

#trainpath1 = '../input/train/*/*'

#trainpath2 = '../input/additional/*/*'



trainpath = '../input/train/*/*'

testpath = '../input/test/*'







#train1 = pd.DataFrame([{'path': c_path, 

#                          'image_name': os.path.basename(c_path), 

#                          CAT_COLUMN: os.path.basename(os.path.dirname(c_path))} 

#                            for c_path in glob(trainpath1)])



#train2 = pd.DataFrame([{'path': c_path, 

#                          'image_name': os.path.basename(c_path), 

#                          CAT_COLUMN: os.path.basename(os.path.dirname(c_path))} 

#                            for c_path in glob(trainpath2)])



#train = train1.append(train2, ignore_index=True)

#train = train1



#del train1, train2



train = pd.DataFrame([{'path': c_path, 

                       'image_name': os.path.basename(c_path), 

                       CAT_COLUMN: os.path.basename(os.path.dirname(c_path))} 

                            for c_path in glob(trainpath)])



print(len(train))



test = pd.DataFrame([dict(path = c_path, 

                          image_name = os.path.basename(c_path)) 

                             for c_path in glob(testpath)])



print(len(test))



train_data = normalize_image_features(train['path'])



le = LabelEncoder()

train_target = le.fit_transform(train['type'].values)



test_data = normalize_image_features(test['path'])



test_id = test.image_name.values
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=input_shape))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))

 

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D((2, 2), strides=(2, 2))) 



model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(MaxPooling2D((2, 2), strides=(2, 2))) 





model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))



model.layers.pop()

model.add(Dense(3, activation='softmax'))







lrate = 0.001

decay = lrate/epochs





opt = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)





model.compile(loss='sparse_categorical_crossentropy',

              optimizer=opt,

              metrics=['accuracy'])





train_datagen = ImageDataGenerator(zca_whitening = True)



train_datagen.fit(train_data)



train_generator = train_datagen.flow(

        x = train_data,

        y = train_target,        

        batch_size=batch_size,

        #class_mode='sparse',

        shuffle = True)



model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs)
test_datagen = ImageDataGenerator(zca_whitening = True)



test_generator = test_datagen.flow(

        x = test_data,              

        batch_size=batch_size,

        class_mode='sparse',

        shuffle = True)



predict = model.predict_generator(generator = test_generator, 

                                  steps = 512)



prediction = pd.DataFrame(test_generator.filenames, columns=['image_name'])

prediction['image_name'] = prediction['image_name'].map(lambda x: x.lstrip('test\\'))

prediction = pd.concat([prediction, pd.DataFrame(predict, columns=['Type_1', 'Type_2', 'Type_3'])], axis=1).to_csv('../output/MobileODTPredict_v5.csv', index=False)
