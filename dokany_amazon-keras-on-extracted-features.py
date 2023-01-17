# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import plotly.plotly as py

import plotly.tools as tls

import cv2

from sklearn.feature_extraction.text import CountVectorizer

from operator import itemgetter

from sklearn import preprocessing





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print(check_output(["ls"]).decode("utf8"))
df = pd.read_csv('../input/train_v2.csv')

print(df.head())



readfromCSVs = False
if readfromCSVs == True:

    X_train = pd.read_csv('X_train.csv')

    Y_train = pd.read_csv('Y_train.csv')

    test_pred = pd.read_csv('pred.csv')

    

    X_train = np.nan_to_num(np.array(X_train))

    Y_train = np.nan_to_num(np.array(Y_train))

    test_pred = np.nan_to_num(np.array(test_pred))

    

    X_train = np.around(X_train, decimals = 3)

    #Y_train = np.around(X_train, decimals = 3)

    test_pred = np.around(test_pred, decimals = 3)

    

    X_train = np.delete(X_train, 0,1)

    Y_train = np.delete(Y_train, 0,1)

    test_pred = np.delete(test_pred, 0,1)



if readfromCSVs == False:

    

    labels = np.array(df['tags'])

    vect = CountVectorizer()

    vect.fit(labels)

    vect.get_feature_names()

    labels_dtm = vect.transform(labels)

    df_labels = pd.DataFrame(labels_dtm.toarray(), columns = vect.get_feature_names())

    

    # create a dict to collect total values of each class of label

    amazon_condition = {}

    for col in df_labels.columns.values:

        z = df_labels.groupby([col])[col].count().astype(int)

        amazon_condition[col] = 0

        for i, j in enumerate(z):

            if i != 0:

                amazon_condition[col] += j

    amazon_condition_labels = [x for x in amazon_condition.keys()]

    amazon_condition_values = [x for x in amazon_condition.values()]



    print(amazon_condition_labels)

    print(amazon_condition_values)
#from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import fbeta_score

from PIL import Image

from PIL import ImageStat

import glob



def extract_features(path):

    features = []

    image_stat = ImageStat.Stat(Image.open(path))

    features += image_stat.sum

    features += image_stat.mean

    features += image_stat.rms

    features += image_stat.var

    features += image_stat.stddev

    img = cv2.imread(path)

    cv2img = cv2.imread(path,0)

    features += list(cv2.calcHist([cv2img],[0],None,[256],[0,256]).flatten())

    mean, std = cv2.meanStdDev(img)

    features += list(mean)

    features += list(std)

    return features
from tqdm import tqdm

from time import time



if readfromCSVs == False:

    X_train = pd.DataFrame()

    input_path = '../input/'

    df['path'] = df['image_name'].map(lambda x: input_path + 'train-jpg/' + x + '.jpg')



    f_list = []



    t0 = time()

    for i in df['path']:

        f = np.array(extract_features(i)).astype(int)

        f_list.append(f)



    print("done in %0.3fs" % (time() - t0))    
if readfromCSVs == False:

    f_list_arr = np.array(f_list)

    X_train = pd.DataFrame(f_list_arr)

    #normalize the X_train scale

    #X_train = preprocessing.scale(X_train)

    for i in X_train.columns.values:

        X_train[i] = X_train[i]/max(X_train[i])

    X_train = np.array(X_train)    

    print(type(X_train))

    #print(X_train.head())
if readfromCSVs == False:

    print(type(X_train))

    print(type(df_labels))

    Y_train = np.array(df_labels)

    #print(Y_train[:5])

    #print(len(X_train[0]))

    pad = np.zeros((len(X_train),42))

    print(pad.shape)

    X_train = np.hstack((X_train,pad))

    print(X_train.shape)




if readfromCSVs == False:

    print(df_labels.head())
#print(test_images)
if readfromCSVs == False:



    test_images = []

    X_test = []

    test_images = glob.glob(input_path + 'test-jpg-v2/*')

    X_test = pd.DataFrame([[x.split('/')[3].replace('.jpg',''),x] for x in test_images])

    print(X_test[:5])



    X_test.columns = ['image_name','path']

    print(X_test[:5])

    print(X_test.shape, type(X_test))
if readfromCSVs == False:

    ftr_list_arr=[]

    ftr_list = []

    pad=[]



    t0 = time()

    for i in X_test['path']:

        ftr = np.array(extract_features(i)).astype(int)

        ftr_list.append(ftr)

    print("done in %0.3fs" % (time() - t0))    



    ftr_list_arr = np.array(ftr_list)
if readfromCSVs == False:

    print(ftr_list_arr.shape)

    print(len(ftr_list_arr))

    print(type(ftr_list_arr))
if readfromCSVs == False:

    pad = np.zeros((len(ftr_list_arr),42))

    print(pad.shape)

    ftr_list_arr = np.hstack((ftr_list_arr,pad))

    test_pred = pd.DataFrame(ftr_list_arr)



    print(test_pred.shape)

    #print(test_pred.head())



    #normalize the test_pred scale

    #test_pred = preprocessing.scale(test_pred)



    for i in test_pred.columns.values:

        test_pred[i] = test_pred[i]/max(test_pred[i])



    test_pred = np.array(test_pred)    
if readfromCSVs == False:

    np.nan_to_num(X_train)

    np.nan_to_num(test_pred)
if readfromCSVs == False:

    X_train_to_csv = pd.DataFrame(X_train)

    Y_train_to_csv = pd.DataFrame(Y_train)

    test_pred_to_csv = pd.DataFrame(test_pred)



    X_train_to_csv.to_csv('X_train.csv')

    Y_train_to_csv.to_csv('Y_train.csv')

    test_pred_to_csv.to_csv('pred.csv')
if readfromCSVs == False:

    print(check_output(["ls"]).decode("utf8"))

    print(check_output(["ls", "../input"]).decode("utf8"))
print(X_train.shape)

print(Y_train.shape)

print(test_pred.shape)
#test_pred = np.array(test_pred)

#print(test_pred.shape)
X_val = X_train[32000:]

X_train = X_train[:32000]

Y_val = Y_train[32000:]

Y_train = Y_train[:32000]
print(Y_train.shape)

print(Y_val.shape)

print(X_train.shape)

print(X_val.shape)

print(test_pred.shape)
from keras import backend as K

img_rows = 18 

img_cols = 18



if K.image_dim_ordering() == 'th':

    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)

    test_pred = test_pred.reshape(test_pred.shape[0], 1, img_rows, img_cols)

    X_val = X_val.reshape(X_val.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

    test_pred = test_pred.reshape(test_pred.shape[0], img_rows, img_cols, 1)

    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)
print(Y_train.shape)

print(Y_val.shape)

print(X_train.shape)

print(X_val.shape)

print(test_pred.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils





batch_size = 128

nb_classes = 17

#nb_epoch = 5

nb_epoch = 10



# input image dimensions

img_rows, img_cols = 18, 18

# number of convolutional filters to use

nb_filters = 32

# size of pooling area for max pooling

pool_size = (2, 2)

# convolution kernel size

kernel_size = (3, 3)



model = Sequential()



model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))

model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=pool_size))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(nb_classes))

#model.add(Activation('softmax'))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])

#model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])



#callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0)]

    

model.fit(np.nan_to_num(X_train), Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, 

          validation_data=(np.nan_to_num(X_val), Y_val), shuffle=True)



#score = model.evaluate(np.nan_to_num(X_val), Y_val, verbose=0)

#print('Test score:', score[0])

#print('Test accuracy:', score[1])
score = model.evaluate(np.nan_to_num(X_val), Y_val, verbose=0)

print('Test score:', score[0])

print('Test accuracy:', score[1])

print(score)
from sklearn.metrics import fbeta_score



#p_valid = model.predict(X_val, batch_size=128)

p_pred = model.predict(np.nan_to_num(X_val), batch_size=128)

#print(y_valid)

#print(p_valid)

#print(fbeta_score(Y_val, np.array(p_valid) > 0.2, beta=2, average='samples'))
p_proba = model.predict_proba(np.nan_to_num(X_val), batch_size=128)
p = model.predict_classes(np.nan_to_num(X_val), batch_size=128)
print(p_pred[:2])

print(p_proba[:2])
col1 = np.array(p_proba[:10][0].sort)



#print(col1[:10])
print(type(col1))

print(col1.shape)

#print(Y_val[:][10].sort)

print(p_proba[:10][0])

print(p_proba.shape)

print(p_proba[:][:10])
print(np.around(p_proba[:10], decimals = 3))

print(Y_val[:10])
#print(X_val[:2])

print(p_proba[:2])

print(Y_val[:2])
print(np.nan_to_num(X_train[0]))
print(type(test_pred))

predset = np.array(test_pred)

predset = predset.reshape(predset.shape[0],18,18,1)
print(predset.shape)
p_test = model.predict(predset, batch_size = 128, verbose=2)
print(p_proba[:1])

print(Y_val[:1])
import os



def load_image(filename, resize, folder):

    img = mpimg.imread('../input/{}/{}.jpg'.format(folder, filename))

    if resize:

        img = cv2.resize(img, (64, 64))

    return np.array(img)



#result [result >0.24] = 1

#result [result < 1] = 0

#print(result[:5])

X_test = os.listdir(input_path + 'test-jpg')

X_test = [fn.replace('.jpg', '') for fn in X_test]



print(X_test)



result = []

TEST_BATCH = 128

for i in range(2, len(X_test), TEST_BATCH):

    X_batch = X_test[i:i+TEST_BATCH]

    X_batch = np.array([load_image(fn, True, 'test-jpg') for fn in X_batch])

    p = model.predict(X_batch)

    result.append(p)

    

print (result)

r = np.concatenate(result)

r = r > 0.5



table = []

for row in r:

    t = []

    for b, v in zip(row, tag_columns):

        if b:

            t.append(v.replace('tag_', ''))

    table.append(' '.join(t))



df_pred = pd.DataFrame.from_dict({'image_name': X_test, 'tags': table})

df_pred.to_csv('submission.csv', index=False)
#print(type(result))

#result_df = pd.DataFrame(result)

#result_df.columns = df_labels.columns.values

#print(result_df.head())

#tags = []



#for i,j in enumerate(np.array(result_df)):

#    temp_tags = []

    #print(temp_tags)

#    for c, col in enumerate(result_df.columns.values):

#        if j[c] == 1:

#            temp_tags.append(col)

#    tags.append(temp_tags)



#tags1 = []    



#for x in tags:

#    st = ''    

#    for y in x:

#        st += y + ' '

#    tags1.append(st[:(len(st)-1)])         
#print(len(tags), type(tags))

#print(tags[:10])

#print(len(tags1), type(tags1))

#print(tags1[:10])

#X_test['tags'] = tags1



#X_test[:10]

#print(X_test.columns.values)
#X_test[['image_name','tags']].to_csv('submission_amazon_02.csv', index=False)
#pdForCSV = pd.DataFrame()

#pdForCSV['image_name'] = X_test.image_name.values

#pdForCSV['tags'] = preds

#pdForCSV.to_csv('2017_05_01_XGB_submission.csv', index=False)