import tensorflow as tf
import keras

import sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import jaccard_score

from scipy import stats

import seaborn as sns

import skimage
from skimage.transform import rotate

from tqdm import tqdm
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPool2D, UpSampling2D, GlobalMaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.layers import Dense, Dropout, Activation, Reshape, Flatten, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import NASNetMobile, Xception, DenseNet121, MobileNetV2, InceptionV3, InceptionResNetV2, vgg16, resnet50, inception_v3, xception, DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16


from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

import numpy as np
import os
import cv2
import pandas as pd
# import imutils
import random
from PIL import Image
import matplotlib.pyplot as plt

import pickle
DATA_PATH = "../input/thai-mnist-classification/"

TRAIN_PATH = DATA_PATH + "train/"
TEST_PATH = DATA_PATH + "test/"

df1 = pd.read_csv(DATA_PATH + "mnist.train.map.csv")
df1.index = df1['id']

x_resolution = 32
y_resolution = 32
batch_size = 32
batch_size_generator = 32
epoch = 20
earlystopping = 60
nan_index = 10
N = df1.shape[0]
N
from skimage.morphology import convex_hull_image
from skimage.util import invert
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
temp_img = cv2.imread("../input/thai-mnist-classification/train/59937745-b5e4-4f69-aee6-3e43a1381846.png")
temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY)
plt.gray()
plt.imshow(temp_img)
crop_img = convex_crop(invert(temp_img),pad=20)
plt.imshow(crop_img)
print(crop_img.shape)
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    
    return img
plt.imshow(thes_resize(temp_img))
def read_image(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im = thes_resize(im)
    return im
x_s = 6
fig = plt.figure(figsize=(24,16))
for i, path in enumerate(df1['id']):
    if i == x_s**2:
        break
    output = read_image('../input/thai-mnist-classification/train/' + path)
    ax = fig.add_subplot(x_s,x_s,1+i)
    ax.imshow(output)
def load_image_1():
    img_path = os.listdir(TRAIN_PATH)
    label = []
    train_img = np.empty((N, x_resolution, y_resolution), dtype=np.uint8)

    for i, image in enumerate(tqdm(df1['id'])):
        train_img[i] = read_image(TRAIN_PATH +image)
        label.append(to_categorical(df1.category[image], num_classes=10))
    
    return np.array(train_img), np.array(label)

X, Y = load_image_1()
import pickle
pickle.dump((X, Y), open('data_training_32x32.data', 'wb'))
X, Y = pickle.load(open('../input/model-image1/data_training_32x32.data', 'rb'))
for i in X[0].flatten()[:100]:
    if i != 0:
        print(i)
X.shape
X = X / 255.
X = X.reshape(-1,x_resolution,y_resolution,1)
X.shape
for i in X[0].flatten()[:100]:
    if i != 0:
        print(i)
with tf.device('/device:GPU:0'):
    def get_model():
        inputs = Input(shape=(x_resolution, y_resolution, 1))
        
        x = Conv2D(64, kernel_size=(5,5), activation='relu')(inputs)
        x = MaxPool2D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(128, kernel_size=(5,5), activation='relu')(x)
        x = MaxPool2D(pool_size=2)(x)
        x = BatchNormalization()(x)
        
        x = Flatten()(x)
        x = Dense(128)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.2)(x)
        
        x = Dense(10)(x)
        outputs = Activation('softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model
    
    model_image = get_model()
kfold = KFold(n_splits=10, random_state=42)

np.unique(Y, return_counts=True)
df1['category'][:5]
folder = 'crop_kfold_10_1'
best_model_filename = './model_image_'+folder+'.h5'
best_model_filename_nokfold = './model_image_splt_crop_1.h5'
# x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)


# EarlyStopper = EarlyStopping(patience=earlystopping, verbose=1, monitor='val_accuracy', mode='max')
# Csv_logger = CSVLogger('./save.csv', append=True, separator=';')
# Checkpoint = ModelCheckpoint(best_model_filename_nokfold, verbose=1, monitor='val_accuracy', save_best_only=True, mode='max')

# model_image.fit(x_train, y_train, 
#          validation_data=(x_valid, y_valid),
#          batch_size=batch_size,
#          epochs=300,
#          verbose=1,
#          callbacks=[EarlyStopper, Checkpoint, Csv_logger]
#         )
val_acc = []
for count, (train_index, valid_index) in enumerate(kfold.split(X)):
#     x_train = X[train_index]
#     y_train = Y[train_index]
#     x_valid = X[valid_index]
#     y_valid = Y[valid_index]
    
    model_image.fit(X[train_index], Y[train_index], 
         validation_data=(X[valid_index], Y[valid_index]),
         batch_size=batch_size,
         epochs=epoch,
         verbose=1
        )
    print(count+1)
    score_1 = model_image.evaluate(X[train_index], Y[train_index])
    score_2 = model_image.evaluate(X[valid_index], Y[valid_index])
    print('========= Genearator Model =========')
    print('Train loss :', score_1[0])
    print('Train accuracy :', score_1[1]*100)
    print('Valid loss :', score_2[0])
    print('Valid accuracy :', score_2[1]*100)
    val_acc.append(score_2[1])
    print('\n\n')
    
model_image.save(best_model_filename)
for num, i in enumerate(val_acc):
    print('Fold :', num+1)
    print('Accuracy :', i*100, '%')
    print('=========================================')
    
print('========= Last Genearator Model =========')
results = model_image.evaluate(X, Y)
print('Loss :', results[0])
print('Accuracy :', results[1]*100, '%')
# with tf.device('/device:GPU:0'):
#     for i, image in enumerate(tqdm(df1['id'])):
#         y_pred = model_image.predict(np.array([X[i]]), verbose=0)
#         if np.argmax(y_pred[0]) != np.argmax(Y[i]):
#             s = np.empty((x_resolution,y_resolution,3))
#             s[:, : , 0] = X[i].reshape(x_resolution,y_resolution)
#             s[:, : , 1] = X[i].reshape(x_resolution,y_resolution)
#             s[:, : , 2] = X[i].reshape(x_resolution,y_resolution)
#             plt.imshow(s)
#             print(np.argmax(y_pred[0]) , np.argmax(Y[i]))
#             plt.show()
# df2 = pd.read_csv(DATA_PATH + "train.rules.csv")
# df2.index = df2['id']
# df2 = df2.fillna(nan_index)
# df2
# model_image = load_model('../input/model-image1/model_image_drop_1.h5')
# model_image.summary()
# model_image.predict(np.array([X[0]]))[0]
# def load_data():
#     img_path = os.listdir(TRAIN_PATH)
#     train_label = []
#     train_index = []
#     train_condition = []
    
#     for i, image in enumerate(tqdm(df2['id'])):
#         train_label.append(df2.predict[image])
#         train_index.append([])
        
#         if df2.feature1[image] != nan_index:
#             image1 = read_image(TRAIN_PATH + df2.feature1[image])
#             train_condition.append(to_categorical(np.argmax(model_image.predict(image1.reshape(1,x_resolution,y_resolution,1))[0]), num_classes=11))
#         else:
#             train_condition.append(to_categorical(nan_index, num_classes=11))
            
#         if df2.feature2[image] != nan_index:    
#             image2 = read_image(TRAIN_PATH + df2.feature2[image])
#             train_index[-1].append(np.argmax(model_image.predict(image2.reshape(1,x_resolution,y_resolution,1))[0]))
#         else:
#             train_index[-1].append(nan_index)
            
#         if df2.feature3[image] != nan_index:
#             image3 = read_image(TRAIN_PATH + df2.feature3[image])
#             train_index[-1].append(np.argmax(model_image.predict(image3.reshape(1,x_resolution,y_resolution,1))[0]))
#         else:
#             train_index[-1].append(nan_index)
    
#     return np.array(train_condition), np.array(train_index), np.array(train_label)

# C_mae, X_mae, Y_mae = load_data()
# X_mae[:10]
# import pickle
# pickle.dump((C_mae, X_mae, Y_mae), open('mae_data_training.data', 'wb'))
# Y_mae[:10]
# C_mae[:10]
# with tf.device('/device:GPU:0'):
#     def get_mae_model():
#         condition = Input(shape=(11,))
#         inputs = Input(shape=(2,))
        
#         y = Dense(64)(condition)
#         y = BatchNormalization()(y)
#         y = Dropout(0.2)(y)
#         y = Dense(1)(y)
#         y = BatchNormalization()(y)
#         y = Dropout(0.2)(y)
        
#         x = concatenate([inputs, y])

#         x = Dense(256)(x)
#         x = Activation('relu')(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
    
#         x = Dense(128)(x)
#         x = Activation('relu')(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
        
#         x = Dense(64)(x)
#         x = Activation('relu')(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
        
#         x = Dense(16)(x)
#         x = Activation('relu')(x)
#         x = BatchNormalization()(x)
#         x = Dropout(0.2)(x)
        
#         x = Dense(1)(x)
#         outputs = Activation('relu')(x)

#         model = Model(inputs=[condition, inputs], outputs=outputs)
#         model.compile(loss='mean_absolute_error', optimizer='adam')
#         model.summary()

#         return model
    
#     model_mae = get_mae_model()
# kfold = KFold(n_splits=10, random_state=42)

# np.unique(Y_mae, return_counts=True)
# Y_mae.shape
# folder = 'drop_1'
# best_model_filename = './model_mae_'+folder+'.h5'

# EarlyStopper = EarlyStopping(patience=earlystopping, verbose=1, monitor='val_loss', mode='min')
# Csv_logger = CSVLogger('./save_mae.csv', append=True, separator=';')
# Checkpoint = ModelCheckpoint(best_model_filename, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
# val_acc = []
# for count, (train_index, valid_index) in enumerate(kfold.split(X_mae)):
# #     x_train = X[train_index]
# #     y_train = Y[train_index]
# #     x_valid = X[valid_index]
# #     y_valid = Y[valid_index]
    
#     model_mae.fit([C_mae[train_index], X_mae[train_index]], Y_mae[train_index], 
#          validation_data=([C_mae[valid_index], X_mae[valid_index]], Y_mae[valid_index]),
#          batch_size=128,
#          epochs=epoch,
#          verbose=1
#         )
#     score = model_mae.evaluate([C_mae[valid_index], X_mae[valid_index]], Y_mae[valid_index])
#     print(count+1)
#     print('========= Genearator Model =========')
#     print('Valid loss :', score)
#     val_acc.append(score)
#     print('\n\n')
    
# model_mae.save(best_model_filename)
# for num, i in enumerate(val_acc):
#     print('Fold :', num+1)
#     print('Loss :', i, '%')
#     print('=========================================')
    
# print('========= Last Genearator Model =========')
# results = model_mae.evaluate([C_mae, X_mae], Y_mae)
# print('Loss :', results)
def fucking_predict(c, x):
    re = 0
    if c == 10:
        re = x[0]+x[1]
    elif c == 0:
        re = x[0]*x[1]
    elif c == 1:
        re = abs(x[0]-x[1])
    elif c == 2:
        re = (x[0]+x[1])*abs(x[0]-x[1])
    elif c == 3:
        re = abs(((x[1]*(x[1]+1)) - x[0]*(x[0]-1))/2)
    elif c == 4:
        re = 50+(x[0]-x[1])
    elif c == 5:
        re = min(x[0], x[1])
    elif c == 6:
        re = max(x[0], x[1])
    elif c == 7:
        re = ((x[0]*x[1])%9)*11
    elif c == 8:
        re = ((x[0]**2)+1)*x[0] + x[1]*(x[1]+1)
        re = re % 99
    elif c == 9:
        re = 50+x[0]
    else:
        print("fuck")
        
    return int(re)
model_image = load_model(best_model_filename_nokfold)
best_model_filename_nokfold
df3 = pd.read_csv(DATA_PATH + "test.rules.csv")
df3.index = df3['id']
df3 = df3.fillna(nan_index)

df_test = pd.read_csv(DATA_PATH + "submit.csv")
image_name =[]
test_image = []
for i, image in enumerate(tqdm(os.listdir(TEST_PATH))):
    image_name.append(image)
    t = read_image(TEST_PATH + image)
    t = t.astype(np.uint8)
    t = t / 255.0
    t = t.reshape(32,32,1)
    test_image.append(t)
    
test_image = np.array(test_image)
test_image.shape
image_name, test_image = pickle.load(open('../input/model-image1/test_data_32x32.data', 'rb'))
label = np.argmax(model_image.predict(test_image), axis=1)
image2number = {'image': image_name, 'category': label}
image2number = pd.DataFrame(image2number)
image2number.index = image2number['image']
image2number
import pickle
pickle.dump((image_name, test_image), open('test_data_32x32.data', 'wb'))
for i, (image, l) in enumerate(zip(test_image, label)):
    plt.title(l, fontsize=64)
    plt.imshow(image.reshape(32,32))
    plt.show()
image2number.category['cb57fbac-b366-4359-9d89-40f048efedaf.png']
def test1():
    N_test = 20000
    img_path = os.listdir(TEST_PATH)
    test_index = np.empty((N_test, 2), dtype=np.uint8)
    test_condition = []
    nan_label = []
    
    data = []
    
    for i, image in enumerate(tqdm(df3['id'])):
        if df3.feature1[image] != nan_index:
            image1 = df3.feature1[image]
            test_condition.append(image2number.category[image1])
        else:
            test_condition.append(nan_index)
            
        if df3.feature2[image] != nan_index:    
            image2 = df3.feature2[image]
            test_index[i][0] = image2number.category[image2]
        else:
            test_index[i][0] = nan_index
            
        if df3.feature3[image] != nan_index:
            image3 = df3.feature3[image]
            test_index[i][1] = image2number.category[image3]
        else:
            test_index[i][1] = nan_index
            
        c = np.argmax(test_condition[-1])
        x1 = test_index[-1][0]
        x2 = test_index[-1][1]
        data.append([c,x1,x2])
        
    return test_condition, np.array(test_index), data

c_label, x_mix, data = test1()

x1 = x_mix[:,0]
x2 = x_mix[:,1]

test_index = np.empty((20000, 2))
for num, (i,j) in enumerate(zip(x1, x2)):
    test_index[num, 0] = i
    test_index[num, 1] = j
test_index
# import pickle
# pickle.dump((con, test_index), open('mae_data_fucking_testing.data', 'wb'))
# def test():
#     img_path = os.listdir(TEST_PATH)
#     test_index = []
#     test_condition = []
    
#     data = []
    
#     for i, image in enumerate(tqdm(df3['id'])):
#         if i == 20:
#             break
#         test_index.append([])
#         if df3.feature1[image] != nan_index:
#             image1 = cv2.imread(TEST_PATH + df3.feature1[image], 0)/255.
#             image1 = cv2.resize(image1, (x_resolution, y_resolution))
#             image1 = image1.astype(np.uint8)
#             test_condition.append(to_categorical(np.argmax(model_image.predict(image1.reshape(1,x_resolution,y_resolution,1))[0])), num_classes=11)
#         else:
#             test_condition.append(to_categorical(nan_index, num_classes=11))
            
#         if df3.feature2[image] != nan_index:    
#             image2 = cv2.imread(TEST_PATH + df3.feature2[image], 0)/255.
#             image2 = cv2.resize(image2, (x_resolution, y_resolution))
#             image2 = image2.astype(np.uint8)
#             test_index[-1].append(np.argmax(model_image.predict(image2.reshape(1,x_resolution,y_resolution,1))[0]))
#         else:
#             test_index[-1].append(nan_index)
            
#         if df3.feature3[image] != nan_index:
#             image3 = cv2.imread(TEST_PATH + df3.feature3[image], 0)/255.
#             image3 = cv2.resize(image3, (x_resolution, y_resolution))
#             image3 = image3.astype(np.uint8)
#             test_index[-1].append(np.argmax(model_image.predict(image3.reshape(1,x_resolution,y_resolution,1))[0]))
#         else:
#             test_index[-1].append(nan_index)
            
#         c = np.argmax(test_condition[-1])
#         x1 = test_index[-1][0]
#         x2 = test_index[-1][1]
#         data.append([c,x1,x2])
#     return np.array([test_condition, test_index]), data
# x_test, data_test = test()
# x_test[1]
# import pickle
# pickle.dump(x_test, open('mae_data_testing.data', 'wb'))
fucking_sub = []
for c,x in zip(c_label,test_index):
    fucking_sub.append(fucking_predict(c,x))

fucking_sub[:20]
# y_sub = model_mae.predict(x_test)
# y_sub
# y_sub = np.argmax(axis=1)
# y_sub
# for (c,x1,x2),y in zip(data, y_sub):
#     print(c, x1, x2, ':',y)
# submission = pd.Dataframe({'id': df3['id'], 'predict': y_sub})
submission = pd.DataFrame({'id': df3['id'], 'predict': fucking_sub})
submission
submission.to_csv('submission_161020_'+folder+'.csv',index=False)
img_path = os.listdir(TEST_PATH)
im = []
full_im = []
for i, image in enumerate(tqdm(img_path)):
    image1 = read_image(TEST_PATH + '/' + image)
    full_im.append(cv2.resize(image1.copy()*255.0, (128,128)))
    im.append(image1)
im = np.array(im)
pickle.dump(im, open('Test_image_accuracy_32x32.data', 'wb'))
import pickle
im = pickle.load(open('Test_image_accuracy_32x32.data', 'rb'))
im = np.array(im)
im = im/255.
np.where(im==1)
im = im.reshape(-1,x_resolution,y_resolution,1)
im.shape
full_im[1]
model_image = load_model(best_model_filename)
best_model_filename
im = np.array(im)
im.shape
pred = np.argmax(model_image.predict(im), axis=1)
pred[:20]
for i, img in enumerate(pred):
    plt.title(img, fontsize=64)
    plt.imshow(full_im[i])
    plt.show()
x_s = 18
fig = plt.figure(figsize=(24,16))
for a in range(10):
    count = 0
    for i, img in enumerate(tqdm(pred)):
        if img == a:
            if count == x_s**2:
                fig = plt.figure(figsize=(24,16))
                count = 0
            ax = fig.add_subplot(x_s,x_s,1+count)
            ax.imshow(full_im[i])
            count += 1