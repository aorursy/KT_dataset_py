import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

import os

import time

import gc

print(os.listdir("../input/"))



import PIL

import cv2
#데이터 path

path = '../input/2019-3rd-ml-month-with-kakr/'

train_img_path = os.path.join(path, 'train')

test_img_path = os.path.join(path, 'test')



#seed

seed = 119
train_df = pd.read_csv(os.path.join(path, 'train.csv'))

test_df = pd.read_csv(os.path.join(path, 'test.csv'))

class_df = pd.read_csv(os.path.join(path, 'class.csv'))

print(train_df.head())
print('Total class is ', len(class_df))

print(class_df.head())
assert len(train_df) == len(os.listdir(train_img_path)) #파일 수와 리스트 수가 동일한지 확인

print('Train Image is : ', len(train_df))



assert len(test_df) == len(os.listdir(test_img_path))

print('Test Image is : ', len(test_df))
#클래스 분포 확인

plt.figure(figsize=(12, 6))

sns.countplot(train_df["class"], order=train_df["class"].value_counts(ascending=True).index)

plt.title("Number of data per each class")

plt.show()
train_df['class'].value_counts().describe()[['min', 'mean', 'max']]
#이미지 출력함수 정의

def plot_img(img_list, path=train_img_path):

    plt.figure(figsize=(20,20))

    

    for num, (index, img) in enumerate(img_list.iterrows()):

        image = cv2.imread(os.path.join(train_img_path, img.img_file))   #image 파일 load

        image = cv2.rectangle(image, (img.bbox_x1, img.bbox_y1), (img.bbox_x2, img.bbox_y2), color=(255,255,0), thickness= 5) #bounding box 생성

        height, width = image.shape[:2] #image의 shape 저장

        plt.subplot(5, 5, num+1) # 10개의 subplot 중 (num+1) 번쨰 subplot 지정

        plt.imshow(image)

        plt.title('%s, (%i * %i)' % (img.img_file, width, height))

        plt.axis('off')

        

img_list = train_df[:10]

plot_img(img_list)
#이미지의 width와 height을 저장하는 함수

def width_height_img(train_df):

    start = time.time()

    w = []  #width를 저장하는 list

    h = []  #height를 저장하는 list



    for num, df in train_df.iterrows():

        image = cv2.imread(os.path.join(train_img_path, df.img_file))

        height, width = image.shape[:2]

        w.append(width)

        h.append(height)



    train_df['width'] = w

    train_df['height'] = h

    train_df['pixel'] = train_df['width'] * train_df['height']

    print(time.time() -start)

    return train_df



train_df = width_height_img(train_df)

print(train_df.head())
plot_img(train_df.loc[train_df.pixel == min(train_df.pixel)])
plot_img(train_df.loc[train_df.pixel == max(train_df.pixel)])
temp = train_df.loc[train_df['class'] == 115]

plot_img(temp[:15])
ratio = train_df['width'] / train_df['height']
plt.hist(ratio, bins=20)

plt.xlim(0.5, 2.5)

plt.ylim(0, 8000)

plt.title('Ratio histogram')

plt.show()
#Bounding box image를 저장하는 함수

def save_cropped_img(train_df=train_df, path=path, save_path = None):

    for num, df in train_df.iterrows():

        img = cv2.imread(os.path.join(path, df.img_file))

        #print(img.shape)

        img = img[df.bbox_y1 : df.bbox_y2, df.bbox_x1: df.bbox_x2]

        #print(img.shape)

        name = save_path + df.img_file

        cv2.imwrite(name, img)
tt = train_df[:1]



for num, df in tt.iterrows():

    print(df.img_file)

    img = cv2.imread(os.path.join(train_img_path, df.img_file))

    print(img.shape)

    img = img[df.bbox_y1 : df.bbox_y2, df.bbox_x1: df.bbox_x2]

    plt.imshow(img)
#새로 만든 이미지를 저장할 path를 생성

'''%%time

!mkdir /crop_train

#crop_train_path = '/crop_train/'

save_cropped_img(train_df, path=train_img_path, save_path=crop_train_path)'''
'''#모든 이미지가 복사됬는지 확인

assert len(os.listdir(train_img_path)) == len(os.listdir(crop_train_path))

print(len(os.listdir(crop_train_path)))'''
#Test 이미지도 저장합니다.

'''%%time

!mkdir /crop_test

crop_test_path = '/crop_test/'

save_cropped_img(test_df, path=test_img_path, save_path=crop_test_path)'''
crop_train_path = '../input/2019-3rd-ml-month-with-kakr/train/'

assert len(os.listdir('../input/crop-image/train_crop/')) == len(os.listdir(crop_train_path))

print('Train images : ', len(os.listdir('../input/crop-image/train_crop/')))



crop_test_path = '../input/2019-3rd-ml-month-with-kakr/test/'

assert len(os.listdir('../input/crop-image/test_crop//')) == len(os.listdir(crop_test_path))

print("Test images : ", len(os.listdir('../input/crop-image/test_crop//')))
#'class' 가 int64 type으로 저장되어 있습니다.

train_df.info()
#type 변환

train_df['class'] = train_df['class'].astype('str')
from sklearn.model_selection import train_test_split



tr_data, val_data = train_test_split(train_df[['img_file', 'class']], train_size=0.9, random_state=seed, stratify=train_df['class'])

test_data = test_df[['img_file']]

print(len(tr_data), len(val_data))
assert tr_data['class'].nunique() == val_data['class'].nunique()

print('All class image in val_data')

sns.countplot(val_data['class'])
from keras.applications.mobilenet import preprocess_input #사용할 모델명

from keras.preprocessing.image import ImageDataGenerator
#Hyper Parameter

input_size = (224,224) 

batch_size = 32

epochs = 22
#preprocessing만 변경해서 사용할 수 있도록 함수로 정의

def make_generator(input_size= input_size, batch_size=batch_size, preprocessing_function=preprocess_input):

    #ImageDataGenerator 설정을 정의합니다.

    train_datagen = ImageDataGenerator(horizontal_flip=True,    #수평 반전

                                       zoom_range= 0.15,        #확대 & 축소

                                       width_shift_range= 0.1,  #수평방향 이동

                                       height_shift_range=0.1,  #수직방향 이동

                                       preprocessing_function= preprocessing_function  

                                      )



    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)



    #Train Generator를 생성합니다.

    train_generator = train_datagen.flow_from_dataframe(dataframe=tr_data, 

                                                        directory=crop_train_path,  #path는 맞게 수정

                                                        x_col='img_file', 

                                                        y_col='class', 

                                                        target_size=input_size, 

                                                        color_mode='rgb',  #RGB라면 'rgb'

                                                        class_mode='categorical', 

                                                        batch_size=batch_size, 

                                                        #shuffe = True,

                                                        seed=seed)



    #Valid Generator를 생성합니다.

    valid_generator = val_datagen.flow_from_dataframe(dataframe=val_data, 

                                                     directory=crop_train_path, #path는 맞게 수정

                                                     x_col= 'img_file',

                                                     y_col= 'class',

                                                     target_size= input_size,

                                                     color_mode= 'rgb',

                                                     class_mode= 'categorical',

                                                     batch_size= batch_size,

                                                     shuffle=True,

                                                     seed = seed)



    #Test Generator를 생성합니다.

    test_generator = test_datagen.flow_from_dataframe(dataframe=test_data, 

                                                        directory=crop_test_path, #path는 맞게 수정

                                                        x_col='img_file',

                                                        y_col=None,              #없는 데이터이므로 None

                                                        target_size=input_size, 

                                                        color_mode='rgb', 

                                                        class_mode=None, 

                                                        batch_size=batch_size,

                                                        shuffle=False,

                                                        seed=seed)

    

    return train_generator, valid_generator, test_generator
train_generator, valid_generator, test_generator = make_generator()
from keras.applications import MobileNet

from keras.layers import Dense, GlobalAveragePooling2D #, ZeroPadding2D, Conv2D

from keras.models import Model

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
#기본 모델을 불러옵니다.

base_model = MobileNet(input_shape=(224,224,3), 

                         weights='imagenet', #imagenet pretrained weight 불러오기

                         include_top=False) #include_top로 끝부분 미포함
#모델의 마지막 부분을 완성합니다. fully_connected layer인 Dense안에는 분류하고자하는 전체 class 수의 넣습니다.

#activation은 class의 수가 여러개(multi) 일떄는 softmax를, 이진분류(binary)인 경우는 sigmoid를 사용합니다.

x = base_model.output

x = GlobalAveragePooling2D()(x)

pred = Dense(196, activation='softmax', kernel_initializer='he_normal')(x)
#최종 모델 생성

model = Model(inputs=base_model.input, output=pred)
'''for layer in base_model.layers:

    layer.trainable = False

    

#마지막 Dense layer의 파라미터만 학습됩니다.

print(model.trainable_weights)'''
model.compile(optimizer=Adam(lr=0.001, epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
#steps_per_epoch 설정 함수

def get_steps(num_samples, batch_size):

    if (num_samples % batch_size) > 0 :

        return (num_samples // batch_size) + 1

    else :

        return num_samples // batch_size
#callback 설정

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3,

                                           verbose=1, factor=0.5, min_lr=0.0001)

filepath = 'model_{val_acc:.2f}_{val_loss:.2f}.h5'

model_ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

#es = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')



callbacks = [learning_rate_reduction, model_ckpt]
#history로 학습과정 데이터를 저장합니다.

history = model.fit_generator(train_generator, 

                              steps_per_epoch=get_steps(len(tr_data), batch_size), 

                              validation_data= valid_generator,

                              validation_steps= get_steps(len(val_data), batch_size),

                              epochs=epochs,

                              callbacks=callbacks, 

                              verbose=1)

gc.collect()
# Plot training & validation accuracy values

fig, ax = plt.subplots()

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
test_generator.reset()    #Generator 초기화

prediction = model.predict_generator(

    generator = test_generator,

    steps = get_steps(len(test_data), batch_size),

    verbose=1

)
predicted_class_indices=np.argmax(prediction, axis=1)



# Generator class dictionary mapping

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

submission["class"] = predictions

submission.to_csv("submission.csv", index=False)

submission.head()
from keras.applications.xception import Xception

from keras.applications.xception import preprocess_input
base_model_2 = Xception(input_shape=(224,224,3), weights= 'imagenet', include_top=False) 

x = base_model_2.output

x = GlobalAveragePooling2D()(x)

pred = Dense(196, activation='softmax', kernel_initializer='he_normal')(x)



model_2 = Model(inputs=base_model_2.input, output=pred)

model_2.compile(optimizer=Adam(lr=0.001, epsilon=1e-08), loss='categorical_crossentropy', metrics=['acc'])
#callback 설정

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3,

                                           verbose=1, factor=0.5, min_lr=0.0001)

filepath = 'model_2_{val_acc:.2f}_{val_loss:.2f}.h5'

model_ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

#es = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')



callbacks = [learning_rate_reduction, model_ckpt]
from random import randint



history_2 = []

for i in range(6):

    seed = randint(0, 200)

    tr_data, val_data = train_test_split(train_df[['img_file', 'class']], train_size=0.9, random_state=seed, stratify=train_df['class'])

    train_generator, valid_generator, test_generator = make_generator(preprocessing_function=preprocess_input)

    

    history = model_2.fit_generator(train_generator, 

                                      steps_per_epoch=get_steps(len(tr_data), batch_size), 

                                      validation_data= valid_generator,

                                      validation_steps= get_steps(len(val_data), batch_size),

                                      epochs=4,

                                      callbacks=callbacks, 

                                      verbose=1)

    history_2.append(history)
acc_list = []

val_acc_list = []

for _, i in enumerate(history_2):

    acc_list.append(i.history['acc'])

    val_acc_list.append(i.history['val_acc'])
fig, ax = plt.subplots()

plt.plot(acc_list)

plt.plot(val_acc_list)

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'val'], loc='upper left')

plt.show()
test_generator.reset()

prediction = model_2.predict_generator(

    generator = test_generator,

    steps = get_steps(len(test_df), batch_size),

    verbose=1

)
predicted_class_indices=np.argmax(prediction, axis=1)



# Generator class dictionary mapping

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

submission["class"] = predictions

submission.to_csv("submission2.csv", index=False)

submission.head()
from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input
def gray_preprocess(img):

    img = preprocess_input(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = np.repeat(img[:,:, np.newaxis], 3, axis=2)

    return img
seed = 119

tr_data, val_data = train_test_split(train_df[['img_file', 'class']], train_size=0.9, random_state=seed, stratify=train_df['class'])
train_generator, valid_generator, test_generator = make_generator(preprocessing_function=gray_preprocess)
for i in train_generator:

    print(i[0][0].shape)

    plt.imshow(i[0][0])

    break
base_model_3 = MobileNet(input_shape=(224,224,3), weights='imagenet', include_top=False)



x = base_model_3.output

x = GlobalAveragePooling2D()(x)

pred = Dense(196, activation='softmax')(x)



model_3 = Model(inputs=base_model_3.input, output=pred)

model_3.compile(optimizer=Adam(lr=0.001, epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy'])
#callback 설정

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3,

                                           verbose=1, factor=0.5, min_lr=0.0001)

filepath = 'model_3_{val_acc:.2f}_{val_loss:.2f}.h5'

model_ckpt = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

#es = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto')



callbacks = [learning_rate_reduction, model_ckpt]
#history로 학습과정 데이터를 저장합니다.

history_3 = model_3.fit_generator(train_generator, 

                              steps_per_epoch=get_steps(len(tr_data), batch_size), 

                              validation_data= valid_generator,

                              validation_steps= get_steps(len(val_data), batch_size),

                              epochs=epochs,

                              callbacks=callbacks, 

                              verbose=1)
# Plot training & validation accuracy values

fig, ax = plt.subplots()

plt.plot(history_3.history['acc'])

plt.plot(history_3.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Val'], loc='upper left')

plt.show()
test_generator.reset()

prediction = model_3.predict_generator(

    generator = test_generator,

    steps = get_steps(len(test_df), batch_size),

    verbose=1

)
predicted_class_indices=np.argmax(prediction, axis=1)



# Generator class dictionary mapping

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]



submission = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

submission["class"] = predictions

submission.to_csv("submission3.csv", index=False)

submission.head()