import os

import pandas as pd

import matplotlib.pyplot as plt

import ast

import numpy as np

import tensorflow as tf

from tqdm import tqdm

from glob import glob

from dask import bag

import cv2

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, DepthwiseConv2D, BatchNormalization, ZeroPadding2D, Lambda

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, AveragePooling2D , Layer

from keras import optimizers

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.metrics import top_k_categorical_accuracy
import os

# cwd = os.getcwd()

file_path = r"../input/quickdraw-doodle-recognition/train_simplified"

files = os.listdir(file_path)

word_category = [f.split(".")[0] for f in files]

chunk_size = 100



# for index, word in enumerate(word_category):

#     df = pd.read_csv(os.path.join(file_path, str(word+".csv")))

#     for k in range(chunk_size):

#         filename = 'train_{}.csv'.format(k)

#         df['file_index'] = index

#         df['cv'] = (df.key_id // 10 ** 7) % chunk_size

#         chunk = df[df.cv == k]

#         chunk = chunk.drop(['key_id'], axis=1)



# print("===Data shuffle Finished===")
# print("===Data Compression Starting===")    

# for k in tqdm(range(chunk_size)):

#     filename = 'train_{}.csv'.format(k)

#     if os.path.exists(filename):

#         df = pd.read_csv(filename)

#         df['rnd'] = np.random.rand(len(df))

#         df = df.sort_values(by='rnd').drop('rnd', axis=1)

#         df.to_csv(filename + '.gz', compression='gzip', index=False)

#         os.remove(filename)

# print("===Data Compression Done==")
from PIL import ImageDraw, Image

def make_img(img_arr) :

    image = Image.new("P", (256,256), color=255)

    image_draw = ImageDraw.Draw(image)

    for stroke in img_arr:

        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 

                             stroke[1][i],

                             stroke[0][i+1], 

                             stroke[1][i+1]],

                            fill=0, width=5)

    return image





def stroke_to_img(strokes): 

    img=np.zeros((256,256))

    for each in ast.literal_eval(strokes):

        for i in range(len(each[0])-1):

            cv2.line(img,(each[0][i],each[1][i]),(each[0][i+1],each[1][i+1]),255,5)

    img=cv2.resize(img,(32,32))

    img=img/255

    return img
one_hot_encoding = np.eye(len(word_category))

category_y_label = dict()

index = 0

for i in word_category:

    category_y_label[i]=one_hot_encoding[index]

    index+=1

    

# f = "train_1.csv.gz"

f = os.path.join(file_path,"horse.csv")

df = pd.read_csv(f)

X = []

Y = []

num=0

# for i in df.values:

#     if i[2]==True:   #recognized가 True일때 

# #         x = make_img(ast.literal_eval(i[1]))

# #         x = np.array(x.resize((64,64)))

#         x = ast.literal_eval(i[1])

#         X.append(x)

#         Y.append(i[4])

#         num+=1

#         if n%1000 == 0:

#             print("==={}번째 완료".format(num))

for i in df.values:

    if i[3]==True:   #recognized가 True일때 

#         x = make_img(ast.literal_eval(i[1]))

#         x = np.array(x.resize((64,64)))

        x = ast.literal_eval(i[1])

        X.append(x)

        Y.append(i[5])

        num+=1

        if num%10000 == 0:

            print("==={}번째 완료".format(num))
X2 =[]

n=0

for i in X:

    x = make_img(i)

    x = np.array(x.resize((64,64)))

    X2.append(x)

    if n%10000 == 0:

        print("==={}번째 완료==".format(n))

    n+=1

X2 = np.array(X2)
Y2 = []

n = 0

for y in Y:

    Y2.append(category_y_label[y])

    if n%10000 == 0:

        print("==={}번째 완료==".format(n))

    n+=1

Y2 = np.array(Y2)
index = 340



for key,value in category_y_label.items():

    if str(Y2[index])==str(value):

        print(key)



for x,y in X[index]:

    plt.plot(x, -np.array(y), lw=3)
## extra imports to set GPU options

import tensorflow as tf

from keras import backend as k

 

###################################

# TensorFlow wizardry

config = tf.compat.v1.ConfigProto()

 

# Don't pre-allocate memory; allocate as-needed

config.gpu_options.allow_growth = True

 

# Only allow a total of half the GPU memory to be allocated

config.gpu_options.per_process_gpu_memory_fraction = 0.5

 

# Create a session with the above options specified.

tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# k.backend.set_session(tf.compat.v1.Session(config=config))

# k.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))

###################################
train_grand=[]

num_class = 340

per_class=2000
class_paths = glob('/kaggle/input/quickdraw-doodle-recognition/train_simplified/*.csv')

for i , c in enumerate(tqdm(class_paths[0:num_class])): 

    train=pd.read_csv(c,usecols=['drawing','recognized'],nrows=per_class*2)

    train=train[train.recognized==True].head(per_class)

    imagebag=bag.from_sequence(train.drawing.values).map(stroke_to_img)

    train_array=np.array(imagebag.compute())

    train_array=np.reshape(train_array,(per_class,-1))    

    label_array=np.full((train.shape[0],1),i)

    train_array=np.concatenate((label_array,train_array),axis=1)

    train_grand.append(train_array)

del train_array

del label_array
train_grand=np.array([train_grand.pop() for i in np.arange(num_class)]) 

height = 32

width = 32

train_grand=train_grand.reshape((-1,(height*width+1))) 

print(train_grand)
specific = 0.1 

sequence_length = 50

cut = int(specific * train_grand.shape[0])

print(cut)



np.random.shuffle(train_grand)

y_train, X_train = train_grand[cut: , 0], train_grand[cut: , 1:]

y_val, X_val = train_grand[0:cut, 0], train_grand[0:cut, 1:]



del train_grand



x_train=X_train.reshape(X_train.shape[0],height,width,1)

x_val=X_val.reshape(X_val.shape[0],height,width,1)



print(y_train.shape, "\n",

      x_train.shape, "\n",

      y_val.shape, "\n",

      x_val.shape)
model = Sequential()

model.add(Conv2D(32, (3, 3), strides=(1, 1), input_shape=(32, 32,1)))

model.add(Activation('relu'))

# model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(32, (3, 3), strides=(1, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),data_format="channels_last"))

model.add(Conv2D(64, (3, 3), strides=(1, 1)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(64, (3, 3), strides=(1, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2),data_format="channels_last"))

model.add(Conv2D(128, (3, 3), strides=(1, 1)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(128, (3, 3), strides=(1, 1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_class))

model.add(Activation('softmax'))

# model.compile(RMSprop(lr=self.learningRate), 'MSE')

model.summary()
def top_3_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)
reduceLROnPlat=ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=3,

                                 verbose=1,mode='auto',min_delta=0.005,

                                 cooldown=5,min_lr=0.0001)



callbacks=[reduceLROnPlat]



model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',

              metrics=['accuracy',top_3_accuracy])



history=model.fit(x=x_train,y=y_train,batch_size=32,epochs=200,

                  validation_data=(x_val,y_val),callbacks=callbacks,verbose=1)
acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

loss= history.history['loss']

val_loss=history.history['val_loss']



epochs=range(1,len(acc)+1)



plt.plot(epochs,acc,label='Training acc')

plt.plot(epochs,val_acc,label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs,loss,label='Training loss')

plt.plot(epochs,val_loss,label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
#%% get test set

ttvlist = []

reader = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv', index_col=['key_id'],

    chunksize=2048)

for chunk in tqdm(reader, total=55):

    imagebag = bag.from_sequence(chunk.drawing.values).map(stroke_to_img)

    testarray = np.array(imagebag.compute())

    testarray = np.reshape(testarray, (testarray.shape[0], 32, 32, 1))

    testpreds = model.predict(testarray, verbose=0)

    ttvs = np.argsort(-testpreds)[:, 0:3]  # top 3

    ttvlist.append(ttvs)

    

ttvarray = np.concatenate(ttvlist)
numstonames={i : v[:-4].replace(' ','_') for i , v in enumerate(os.listdir(file_path))}

preds_df = pd.DataFrame({'first': ttvarray[:,0], 'second': ttvarray[:,1], 'third': ttvarray[:,2]})

preds_df = preds_df.replace(numstonames)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']



result = pd.read_csv('../input/quickdraw-doodle-recognition/sample_submission.csv', index_col=['key_id'])

result['word'] = preds_df.words.values

result.to_csv('submission.csv')

result.head()
df2 = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv', nrows=100)

df2['drawing'] = df2['drawing'].apply(ast.literal_eval)

n = 10

fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))

for i, drawing in enumerate(df2['drawing']):

    ax = axs[i // n, i % n]

    for x, y in drawing:

        ax.plot(x, -np.array(y), lw=3)

plt.savefig("test_img.png")

plt.show();