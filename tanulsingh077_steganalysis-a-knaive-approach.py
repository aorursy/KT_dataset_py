# PRELIMINARIES

import os

import skimage.io as sk

import matplotlib.pyplot as plt

from scipy import spatial

from tqdm import tqdm

from PIL import Image

from random import shuffle





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"

train_imageids = pd.Series(os.listdir(BASE_PATH + '/Cover')).sort_values(ascending=True).reset_index(drop=True)

test_imageids = pd.Series(os.listdir(BASE_PATH + '/Test')).sort_values(ascending=True).reset_index(drop=True)
cover_images_path = pd.Series(BASE_PATH + '/Cover/' + train_imageids ).sort_values(ascending=True)

JMIPOD_images_path = pd.Series(BASE_PATH + '/JMiPOD/'+train_imageids).sort_values(ascending=True)

JUNIWARD_images_path = pd.Series(BASE_PATH + '/JUNIWARD/'+train_imageids).sort_values(ascending=True)

UERD_images_path = pd.Series(BASE_PATH + '/UERD/'+train_imageids).sort_values(ascending=True)

test_images_path = pd.Series(BASE_PATH + '/Test/'+test_imageids).sort_values(ascending=True)

ss = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
final=[]

def create_labels(cover,jmipod,juniward,uerd,image_id):

    image = sk.imread(cover)

    jmipodimg = sk.imread(jmipod)

    juniward = sk.imread(juniward)

    uerd = sk.imread(uerd)

    

    vec1 = np.reshape(image,(512*512*3))

    vec2 = np.reshape(jmipodimg,(512*512*3))

    vec3 = np.reshape(juniward,(512*512*3))

    vec4 = np.reshape(uerd,(512*512*3))

    

    cos1 = spatial.distance.cosine(vec1,vec2)

    cos2 = spatial.distance.cosine(vec1,vec3)

    cos3 = spatial.distance.cosine(vec1,vec4)

    

    final.append({'image_id':image_id,'jmipod':cos1,'juniward':cos2,'uerd':cos3})
for k in tqdm(range(30000)):

    create_labels(cover_images_path[k],JMIPOD_images_path[k],JUNIWARD_images_path[k],UERD_images_path[k],train_imageids[k])
train_temp = pd.DataFrame(final)

train_temp.head()
def sigmoid(X):

   return 1/(1+np.exp(-X))
train_temp['jmipod'] = train_temp['jmipod'].apply(lambda x:sigmoid(x))

train_temp['juniward'] = train_temp['juniward'].apply(lambda x:sigmoid(x))

train_temp['uerd'] = train_temp['uerd'].apply(lambda x:sigmoid(x))
train_temp.head()
IMG_SIZE = 300

def load_training_data():

  train_data = []

  data_paths = [cover_images_path,JUNIWARD_images_path,JMIPOD_images_path,UERD_images_path]

  labels = [np.zeros(train_temp.shape[0]),train_temp['juniward'],train_temp['jmipod'],train_temp['uerd']]

  for i,image_path in enumerate(data_paths):

    for j,img in enumerate(image_path[:10000]):

        label = labels[i][j]

        img = Image.open(img)

        img = img.convert('L')

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

        train_data.append([np.array(img), label])

        

  shuffle(train_data)

  return train_data
def load_test_data():

    test_data = []

    for img in test_images_path:

        img = Image.open(img)

        img = img.convert('L')

        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

        test_data.append([np.array(img)])

            

    return test_data
%%time

train = load_training_data()
len(train)
plt.imshow(train[115][0], cmap = 'gist_gray')
trainImages = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

trainLabels = np.array([i[1] for i in train])
#PRELIMINARIES

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers. normalization import BatchNormalization
model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(BatchNormalization())

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam',metrics = ['mean_squared_error'])
print(model.summary())
model.fit(trainImages, trainLabels, batch_size = 100, epochs = 3, verbose = 1)
%%time

test = load_test_data()

testImages = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
%%time

predict = model.predict(testImages,batch_size=100)
ss['Label'] = predict
ss.to_csv('submission.csv',index=False)
train_temp.to_csv('train.csv',index=False)