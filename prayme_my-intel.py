import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



import os

print(os.listdir("../input"))
test_path = '../input/seg_test/seg_test/'

train_path = '../input/seg_train/seg_train/'

pred_path = '../input/seg_pred/seg_pred'



test_list = os.listdir('../input/seg_test/seg_test/')

print('test list : ', test_list)



train_list = os.listdir('../input/seg_train/seg_train')

print('train list : ',train_list)



# print(os.path.join(test_path,test_list[0]))



# for a in test_list:

#     print(os.path.join(test_path,a))







# dataframe 만드려다가 포기

# tmp = {'id' : os.listdir(os.path.join(test_path, test_list[0])),

#        'label' : 0}

# tmp2 = {'id' : os.listdir(os.path.join(test_path,test_list[1])),

#         'label' : 1}
# dataframe 실패

# test_df = pd.DataFrame(tmp)

# test_df.head()

# label = [0, 1, 2, 3, 4, 5]



pred_list = os.listdir('../input/seg_pred/seg_pred')

print(len(pred_list))
import cv2

import glob

from keras.preprocessing import image

from tqdm import tqdm_notebook as tqdm



#빈 dict 생성

images_per_class = {}



for class_folder_name in train_list: # train_list = ['sea', 'glacier', 'forest', 'street', 'mountain', 'buildings']

    class_folder_path = os.path.join(train_path, class_folder_name) # path.join을 이용해 상위 for loop마다 폴더 안으로 진입한다.

    class_label = class_folder_name # 폴더의 이름이 곧 category, label이 된다.

    images_per_class[class_folder_name] = [] # 빈 list 생성

    for image_path in tqdm(glob.glob(os.path.join(class_folder_path, '*.jpg'))): 

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)  # numpy.ndarray

        images_per_class[class_label].append(image_bgr) 
for key,value in images_per_class.items():

    print('{0} -> {1}'.format(key, len(value)))
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



def plot_for_class(label):

    nb_rows = 3

    nb_cols = 3

    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(6,6))

    

    n = 0

    for i in range(0, nb_rows):

        for j in range(0, nb_cols):

            axs[i, j].xaxis.set_ticklabels([])

            axs[i, j].yaxis.set_ticklabels([])

            axs[i, j].imshow(images_per_class[label][n])

            n += 1
for label in images_per_class.keys():

    plot_for_class(label)
len(images_per_class['buildings'][-1])
print(images_per_class['buildings'][-1])

print('-'*50)

print(image)
# sea -> 2274

# glacier -> 2404

# forest -> 2271

# street -> 2382

# mountain -> 2512

# buildings -> 2191

from keras.applications.vgg19 import preprocess_input

from keras.preprocessing.image import img_to_array

x_data = []

y_data = []





i = 0

# if i is 0:

#    print(image)

#    print(type(image))

    

for idx, images in enumerate(list(images_per_class.values())):

    

    for image in tqdm(images):

        image = img_to_array(image) # float32 로 바뀜 shape = 150, 150, 3

#         image = preprocess_input(image) # 값이 작아지긴 했e 'images_per_class' is not defined는데 어떻게 바뀌는건지 모르겠음

       # if i is 0:

        #    plt.show(image)

        

        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) # shape = 224, 224 , 3

        image /= 255.

        if i is 0:

            plt.show(image)

            i+=1

        

        

        

        x_data.append(image)

        y_data.append([idx])
plt.show()
print(image)
print(image)
print(image)
print(image)
print(image)

plt.imshow(image) # 값이 완전 무작위
plt.imshow(image) # preprocess 대신 /= 255 했을 때 값들도 0 ~ 1사이로 나옴
plt.imshow(image) # -100대
print(type(x_data))

print(len(x_data))

print(len(y_data))
# x_data = np.zeros((len(x_data), 224, 224, 3), np.float32)

x_data = np.array(x_data[0:len(x_data):3], np.float32)
import keras



y_data = np.array(y_data[0:len(y_data):3], np.uint8)

print(y_data[0:10])

print(y_data.shape)

y_data = keras.utils.to_categorical(y_data, num_classes=6)

print(y_data[0:10])

print(y_data.shape)
print(x_data.shape)
from keras.applications.vgg19 import decode_predictions



# base_model.summary()

predict = base_model.predict(x_data)

y_predict = decode_predictions(predict)

print(y_predict)
from keras.applications import VGG19



# create the base pre-trained model

# VGG19(include_top=True, weights='imagenet', input_tensor=None,

# input_shape=None, pooling=None, classes=1000)

base_model = VGG19(include_top=False,pooling='max')

x_train_valid_bf = base_model.predict(x_data, batch_size=32, verbose=1)
from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid_bf, y_data, test_size=0.2)



print(x_train.shape, x_valid.shape)

print(y_train.shape, y_valid.shape)
from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout



model = Sequential([

    Dense(256, activation='relu', input_shape=(512, )),

    Dropout(0.5),

    Dense(6, activation='softmax')

])

model.summary()
from keras.optimizers import Adam



model.compile(optimizer=Adam(lr=0.0001),

              loss='categorical_crossentropy',

              metrics=['accuracy'])
history = model.fit(

    x=x_train,

    y=y_train,

    batch_size=64,

    epochs=50,

    validation_data=(x_valid, y_valid)  # test_data 넣기

)
# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()