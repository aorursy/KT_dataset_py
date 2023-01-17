# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = '/kaggle/input/training'

test_data = '/kaggle/input/test'

validation_data = '/kaggle/input/validation'

label_data = pd.read_csv('/kaggle/input/labels-foldered.csv',header=None)

label_data.rename(columns={0:'filename',1:'status'},inplace=True)

label_data.index += 1

print(label_data.head())

print(label_data.tail())
import cv2

from random import shuffle

from tqdm import tqdm

from keras.models import Sequential,Model

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import *



img_width = 256

img_length = 256



def label(img):

    index = int(img.split('.')[0])

    if(label_data.loc[index,'status']=='yes'):

        lab = np.array([1])

    elif(label_data.loc[index,'status']=='no'):

        lab = np.array([0])

    return lab

    

def train_data_with_label():

    train_images = []

    for i in tqdm(os.listdir(train_data)):

        path = os.path.join(train_data,i)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_width,img_length))

        train_images.append([np.array(img), label(i)])

    shuffle(train_images)

    return train_images



def test_data_with_label():

    test_images = []

    for i in tqdm(os.listdir(test_data)):

        path = os.path.join(test_data,i)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_width,img_length))

        test_images.append([np.array(img), label(i)])

    shuffle(test_images)

    return test_images



def validation_data_with_label():

    validation_images = []

    for i in tqdm(os.listdir(validation_data)):

        path = os.path.join(validation_data,i)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (img_width,img_length))

        validation_images.append([np.array(img), label(i)])

    shuffle(validation_images)

    return validation_images
training_images = train_data_with_label()

testing_images = test_data_with_label()

validation_images = validation_data_with_label()
from keras.utils import to_categorical



tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,img_width,img_length,1)

tr_lbl_data = np.array([i[1] for i in training_images])

tr_lbl_data = to_categorical(tr_lbl_data)



tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,img_width,img_length,1)

tst_lbl_data = np.array([i[1] for i in testing_images])

tst_lbl_data = to_categorical(tst_lbl_data)



val_img_data = np.array([i[0] for i in validation_images]).reshape(-1,img_width,img_length,1)

val_lbl_data = np.array([i[1] for i in validation_images])

val_lbl_data = to_categorical(val_lbl_data)
print("Dimensi training image : ", tr_img_data.shape)

print("Dimensi validation image : ", val_img_data.shape)

print("Dimensi testing image : ", tst_img_data.shape)
tst_lbl_data
model = Sequential()

model.add(InputLayer(input_shape=[img_width,img_length,1]))



model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=5,padding='same'))



model.add(Conv2D(filters=50,kernel_size=5,strides=1,padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=5,padding='same'))



model.add(Conv2D(filters=80,kernel_size=5,strides=1,padding='same',activation='relu'))

model.add(MaxPool2D(pool_size=5,padding='same'))



model.add(Dropout(rate=0.25))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(rate=0.5))

model.add(Dense(2,activation='softmax'))



optimizer = Adam()

model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])



early_stopping_monitor = EarlyStopping(patience=70)

checkpoint = ModelCheckpoint('.mdl_wts.hdf5',monitor='val_accuracy',save_best_only=True, mode='auto')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, shuffle=True, min_delta=1e-1, mode='min')
model.fit(tr_img_data, tr_lbl_data, epochs=500, batch_size=100, validation_data=(val_img_data, val_lbl_data), verbose=2,callbacks=[early_stopping_monitor,checkpoint,reduce_lr_loss])
import matplotlib.pyplot as plt

plt.plot(model.history.history['loss'], label='train')

plt.plot(model.history.history['val_loss'], label='test')

plt.legend()

plt.show()
test_loss, test_acc = model.evaluate(tst_img_data,tst_lbl_data,batch_size=128)

val_loss, val_acc = model.evaluate(val_img_data,val_lbl_data,batch_size=128)

print("Test Accuracy : ",test_acc)

print("Validation Accuracy : ", val_acc)
from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
import seaborn as sns

from sklearn.metrics import confusion_matrix

true = np.argmax(np.concatenate((tst_lbl_data,val_lbl_data)),axis=1)

nn_predict = model.predict_classes(np.concatenate((tst_img_data,val_img_data)))

conf_mat = confusion_matrix(true,nn_predict)

ax = sns.heatmap(conf_mat,annot=True,fmt='g',cmap='coolwarm')

labels = ['Yes','No']

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

ax.set(xlabel='Kenyataan', ylabel='Diprediksi', title='Confusion Matrix pada Klasifikasi Gambar (Neural Network)')

plt.show()
model.summary()
new_model = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)



new_model_train = new_model.predict(tr_img_data)

print(new_model_train.shape)



new_model_val = new_model.predict(val_img_data)

print(new_model_val.shape)



new_model_test = new_model.predict(tst_img_data)

print(new_model_test.shape)
from sklearn.svm import SVC

svm = SVC(kernel='sigmoid')

svm.fit(new_model_train,np.argmax(tr_lbl_data,axis=1))

print("SVM Training Score : ",svm.score(new_model_train,np.argmax(tr_lbl_data,axis=1)))

print("SVM Validation Score : ",svm.score(new_model_val,np.argmax(val_lbl_data,axis=1)))

print("SVM Test Score : ", svm.score(new_model_test,np.argmax(tst_lbl_data,axis=1)))



svm_predict = svm.predict(np.concatenate((new_model_test,new_model_val)))

conf_mat = confusion_matrix(true,svm_predict)

ax = sns.heatmap(conf_mat,annot=True,fmt='g',cmap='coolwarm')

labels = ['Yes','No']

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

ax.set(xlabel='Kenyataan', ylabel='Diprediksi', title='Confusion Matrix pada Klasifikasi Gambar (SVM)')

plt.show()
import time



images_test_nn = np.concatenate((tst_img_data,val_img_data))

start = time.clock() 

nn_predict = model.predict_classes(images_test_nn)

end = time.clock()

print("Waktu Prediksi NN per Gambar: {} ".format((end-start)/len(images_test_nn))) 



images_test_svm = np.concatenate((new_model_test,new_model_val))

start = time.clock() 

svm.predict(images_test_svm)

end = time.clock()

print("Waktu Prediksi SVM per Gambar: {} ".format((end-start)/len(images_test_svm)))
import xgboost as xgb

xb = xgb.XGBClassifier(silent=False, 

                      scale_pos_weight=1,

                      learning_rate=0.01,  

                      colsample_bytree = 0.4,

                      subsample = 0.8, 

                      n_estimators=100, 

                      reg_alpha = 0.3,

                      max_depth=4, 

                      gamma=10)

xb.fit(new_model_train,np.argmax(tr_lbl_data,axis=1))

print("XGBoost Training Score : ",xb.score(new_model_train,np.argmax(tr_lbl_data,axis=1)))

print("XGBoost Validation Score : ",xb.score(new_model_val,np.argmax(val_lbl_data,axis=1)))

print("XGBoost Test Score : ", xb.score(new_model_test,np.argmax(tst_lbl_data,axis=1)))

xgb_predict = xb.predict(np.concatenate((new_model_test,new_model_val)))

conf_mat = confusion_matrix(true,xgb_predict)

ax = sns.heatmap(conf_mat,annot=True,fmt='g',cmap='coolwarm')

labels = ['Yes','No']

ax.set_xticklabels(labels)

ax.set_yticklabels(labels)

ax.set(xlabel='Kenyataan', ylabel='Diprediksi', title='Confusion Matrix pada Klasifikasi Gambar (XGBoost)')

plt.show()
from IPython.display import display, Image

def predict_image_model_1(path):

    test_images_test = []

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Read path, then change to Grayscale

    img = cv2.resize(img, (img_width,img_length)) #Resize image

    test_images_test.append([np.array(img)]) #Place it to array

    tst_img_data = np.array([i[0] for i in test_images_test]).reshape(-1,img_width,img_length,1) #Reshape image dimension

    value = model.predict(tst_img_data)[0][1] #Predict image with Model 1



    if(value > 0.5):

        label = "Banjir"

    else:

        label = "Tidak Banjir"



    display(Image(filename=path)) #Show image

    print("Confidence : ", value) #Print result

    print("Condition :",label)

    

def predict_image_model_2(path):

    test_images_test = []

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Read path, then change to Grayscale

    img = cv2.resize(img, (img_width,img_length)) #Resize image

    test_images_test.append([np.array(img)]) #Place it to array

    tst_img_data = np.array([i[0] for i in test_images_test]).reshape(-1,img_width,img_length,1) #Reshape image dimension

    value_2 = new_model.predict(tst_img_data) #Output of Flatten

    value = svm.predict(value_2)[0] #Predict image with Model 2



    if(value > 0.5):

        label = "Banjir"

    else:

        label = "Tidak Banjir"



    display(Image(filename=path)) #Show image

    print("Confidence : ", value) #Print result

    print("Condition :",label)

    

def predict_image_model_3(path):

    test_images_test = []

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #Read path, then change to Grayscale

    img = cv2.resize(img, (img_width,img_length)) #Resize image

    test_images_test.append([np.array(img)]) #Place it to array

    tst_img_data = np.array([i[0] for i in test_images_test]).reshape(-1,img_width,img_length,1) #Reshape image dimension

    value_2 = new_model.predict(tst_img_data) #Output of Flatten

    value = xb.predict(value_2)[0] #Predict image with Model 3



    if(value > 0.5):

        label = "Banjir"

    else:

        label = "Tidak Banjir"



    display(Image(filename=path)) #Show image

    print("Confidence : ", value) #Print result

    print("Condition :",label)
predict_image_model_1('/kaggle/input/training/1376.jpg')
predict_image_model_2('/kaggle/input/test/105.jpg')
predict_image_model_3('/kaggle/input/validation/267.jpg')