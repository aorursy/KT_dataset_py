#pins face recognition

# Siamese network for face

%matplotlib inline

import re

import numpy as np

from PIL import Image

from keras import models

from skimage import io

from skimage.transform import rescale, resize, downscale_local_mean

import tensorflow as tf

import os as os

import cv2

import pandas as pd

import matplotlib.image as img

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Activation

from tensorflow.keras.regularizers import l2

from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras as keras;

print(keras.__version__)
def read_images(path):

    df = pd.DataFrame()

    images_path = path

    k=0

    for celeb in os.listdir(images_path):

        i=0

        fullpath = os.path.join(images_path,celeb) 

        for celeb_name in os.listdir(fullpath):

            celeb_path = os.path.join(fullpath, celeb_name)

            im = cv2.imread(celeb_path)

            im = cv2.resize(im,(int(224),int(224)))

#             if (im.shape == (299,299,3)):

                

#           im = img.imread(celeb_path)  #reading image using matplotlib

            dic = {'celeb_folder':celeb,'celeb_photo_name':celeb_name,'Image':im}

            df = df.append(dic, ignore_index=True)

            print(im.shape)

        print(celeb)

        k+=1

        print(k)

        print("DEFAULTER IMAGES",i)

#     df.to_csv('Collective_data.csv')

    return df

train = '../input/pins-face-recognition/PINS'

global_df_train = read_images(train)

print(global_df_train)
print(global_df_train['Image'][0].shape)
test =  '../input/pins-face-recognition/PINS_TEST'

global_df_test = read_images(test)

print(global_df_test)
print(global_df_train.head())

print(global_df_test.head())
def get_pairs(df,i):

    positive = pd.DataFrame()

    negative = pd.DataFrame()

    row = df.loc[df['celeb_photo_name'].str.contains(i)]

    for rows in df.iterrows():

        if(i in rows[1][2]):

            A = rows[1][0]

            break

    p_temp = row.sample(n=30).index

    row_negative = df.loc[~df['celeb_photo_name'].str.contains(i)]

    n_temp = row_negative.sample(n=30).index

    for index in p_temp:

        x = df.iloc[index]

        positive = positive.append({'Anchor':A,'Another':x['Image'],'Label':1},ignore_index=True) 

    for index in n_temp:

        x = df.iloc[index]

        negative = negative.append({'Anchor':A,'Another':x['Image'],'Label':0},ignore_index=True)

    return pd.concat([positive,negative],axis = 0, ignore_index=True)

   
L = []

final = pd.DataFrame()

for celeb in os.listdir(train):

    L.append(celeb.split("_")[1])

for i in L:

    P = get_pairs(global_df_train,i)

    final = final.append(P,ignore_index=True)

print(final)

M = []

final_test = pd.DataFrame()

for celeb in os.listdir(test):

    M.append(celeb.split("_")[1])

for i in M:

    P = get_pairs(global_df_test,i)

    final_test = final_test.append(P,ignore_index=True)

print(final_test)
# final['Anchor'][0].reshape(np.newaxis ,299,299)

for row in final.iterrows():

    if ((row[1][0].shape!= (299,299,3)) | (row[1][1].shape != (299,299,3))): 

        print(row[1][0].shape,row[1][1].shape)



   
def build_base_network(input_shape):

    #create model

    img_a = Input(shape = input_shape)

    img_b = Input(shape = input_shape)

    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    #add model layers

#     model.add(Conv2D(16, kernel_size=(3,3),activation='relu',padding='valid', input_shape=input_shape, data_format='channels_last'))

# #     model.add(MaxPooling2D())

# #     model.add(Conv2D(128, kernel_size=(10,10),activation='relu',kernel_regularizer = l2(2e-4)))

# #     model.add(MaxPooling2D())

# #     model.add(Conv2D(256, kernel_size=(7,7),activation='relu',kernel_regularizer = l2(2e-4)))

#     #-----------------------------------------------------------------------------------------------------

#     model.add(MaxPooling2D())

#     model.add(Conv2D(32, kernel_size=(3,3),activation='relu',kernel_regularizer = l2(2e-4)))

#     model.add(MaxPooling2D())

#     model.add(Conv2D(64, kernel_size=(3,3),activation='relu',kernel_regularizer = l2(2e-4)))

#     #---------------------------------------------foraccuracy

# #     model.add(MaxPooling2D())

# #     model.add(Conv2D(256, kernel_size=(4,4), activation='relu'))#,kernel_regularizer = l2(2e-4)))

# #     model.add(MaxPooling2D())

# #     model.add(Conv2D(256, kernel_size=(3,3), activation='relu',kernel_regularizer = l2(2e-4)))

# #     model.add(MaxPooling2D())

# #     model.add(Dropout(0.25))

#     model.add(Flatten())

# #     model.add(Dropout(0.5))

#     model.add(Dense(256, activation='sigmoid')) # kernel_regularizer = l2(1e-3),

    model.summary()

    feat_vecs_a = model(img_a)

    feat_vecs_b = model(img_b)

    layer = Lambda(lambda x: K.abs(x[0] - x[1]))

    distance = layer([feat_vecs_a,feat_vecs_b])

#     distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

    prediction = Dense(1,activation = 'sigmoid')(distance)

    model_siamese = Model(inputs=[img_a, img_b], outputs=prediction)

    model_siamese.summary()

    return model_siamese
def euclidean_distance(vects):

    x, y = vects

    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))





def eucl_dist_output_shape(shapes):

    shape1, shape2 = shapes

    return (shape1[0], 1)
epochs = 15

rms = RMSprop()
def contrastive_loss(y_true, y_pred):

    margin = 1

    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
def accuracy(y_true, y_pred):

    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
base_network = build_base_network((224,224,3))
adam = tf.keras.optimizers.Adam(lr=0.000006)

base_network.compile(loss=contrastive_loss, optimizer=adam, metrics = [accuracy])
final = final.sample(frac=1)

final = final.reset_index(drop=True)

final_test = final_test.sample(frac=1)

final_test = final_test.reset_index(drop=True)
X_train_test = np.stack(final_test['Anchor'])

Y_train_test = np.stack(final_test['Another'])

Label_test = np.vstack(final_test['Label'])
iter1 = int(final.shape[0]/60)

iter2 = int(final_test.shape[0]/32)

for k in range(0,21):

    Q = final

    R = final_test

    L = []

    A = []

    test_L = []

    test_A = []

    G = R.sample(32)

    X_test = np.stack(G['Anchor'])

    Y_test = np.stack(G['Another'])

    Test_Label = np.vstack(G['Label'])

    for i in range(0,iter1):

        F = Q.sample(60,random_state=i)

        Q = Q.drop(F.index)

        X_train = np.stack(F['Anchor'])

        Y_train = np.stack(F['Another'])

        Label = np.vstack(F['Label'])

        Loss, Acc = base_network.train_on_batch([X_train,Y_train],Label)

        L.append(Loss)

        A.append(Acc)

#         print(i,"Loss:",Loss,"Accuracy:",Acc)\

    for j in range(1,iter2):

        S = R.sample(32,random_state=j)

        X_train = np.stack(S['Anchor'])

        Y_train = np.stack(S['Another'])

        Label = np.vstack(S['Label'])

        Loss, Acc = base_network.test_on_batch([X_test,Y_test],Test_Label)

        test_L.append(Loss)

        test_A.append(Acc)

    AVG_LOSS = sum(L)/len(L)

    AVG_ACC = sum(A)/len(A)

    VAL_LOSS = sum(test_L)/len(test_L)

    VAL_ACC = sum(test_A)/len(test_A)

    print(k,") ","Loss:",AVG_LOSS*100, "Accuracy:",AVG_ACC*100)

    print("    Validation Loss:",VAL_LOSS*100, "Validation Accuracy:",VAL_ACC*100)
def predict_label(img1,img2,model):

    label = model.predict([[img1],[img2]])

    if label <= 0.5:

        return 1.0

    else:

        return 0.0



print('-----------------------------------------------')

k = 0

for i in range(0,len(final_test)):

#     print("New Test")

   

    l = final_test['Label'][i]

    j = predict_label(final_test['Anchor'][i],final_test['Another'][i],base_network)

#     print("True Label:{}  Pred Label:{}".format(final_test['Label'][i],predict_label(final_test['Anchor'][i],final_test['Another'][i],base_network)))

    if  l != j:

        k+=1

    print(i)

#     f = plt.figure()

#     f.add_subplot(1,2, 1)

#     plt.imshow(np.rot90(np.rot90(np.rot90(final_test['Anchor'][i],2))))

#     f.add_subplot(1,2, 2)

#     plt.imshow(np.rot90(np.rot90(np.rot90(final_test['Another'][i],2))))

#     plt.show(block=True)

print("Wrong Prediction Count:",k)
print("Wrong Prediction Count:",k)

# base_network.save('75_pins_siamese_net.h5')

tf.saved_model.save(base_network, "./")
# base_network.input

# base_network.output
model = tf.saved_model.load('./')
infer = model.signatures["serving_default"]

print(infer.structured_outputs)
# infer = model.signatures["serving_default"]

# print(infer.structured_input_signature)