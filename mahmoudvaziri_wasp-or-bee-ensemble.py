import numpy as np

import pandas as pd 

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential,load_model

from tqdm import tqdm

import cv2 as cv

import tensorflow as tf

import matplotlib.pyplot as plt

import gc

from sklearn.metrics import accuracy_score

from scipy import stats
main_folder = '../input/bee-vs-wasp/kaggle_bee_vs_wasp/'

df = pd.read_csv('../input/bee-vs-wasp/kaggle_bee_vs_wasp/labels.csv')

df = df[df.photo_quality==1]

df.head()
'''

From https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

'''

for idx in tqdm(df.index):    

    df.loc[idx,'path']=df.loc[idx,'path'].replace('\\', '/') 

    

df.head()
df_test = df[df.is_final_validation==1].reset_index()

df_train = df[df.is_final_validation!=1].reset_index()

df_train.shape,df_test.shape
df.label.value_counts().plot.pie(autopct='%1.1f%%')
'''

From https://www.kaggle.com/koshirosato/bee-or-wasp-base-line-using-resnet50

'''

img_size = 225

def create_datasets(df, img_size):

    imgs = []

    for path in tqdm(df['path']):

        img = cv.imread(main_folder + path)

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        img = cv.resize(img, (img_size,img_size))

        imgs.append(img)

        

    imgs = np.array(imgs, dtype='float32')

    imgs = imgs / 255.0

    df = pd.get_dummies(df['label'])

    return imgs, df





train, df_train = create_datasets(df_train, img_size)

test, df_test = create_datasets(df_test, img_size)
print(train.shape)

sub=[]

for i in range(5):

    sub.append(np.random.choice(train.shape[0], 3000, replace=False))

sub = np.array(sub).T

sub.shape
def mobilenet():

    model = Sequential()

    model.add(layers.Input(shape=(img_size,img_size,3)))

    model.add(tf.keras.applications.MobileNetV2(include_top=False,weights="imagenet"))

    #model.add(layers. GlobalAveragePooling2D())#BatchNormalization()

    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(128,activation='relu'))

    #model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(3,activation='softmax'))

    for layer in model.layers[:1]:

        layer.trainable = False

    #mobilenet.summary()

    return model



def xception():

    model = Sequential()

    model.add(layers.Input(shape=(img_size,img_size,3)))

    model.add(tf.keras.applications.Xception(weights='imagenet',include_top=False))

    model.add(layers. GlobalAveragePooling2D())#BatchNormalization()

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(3,activation='softmax'))

    for layer in model.layers[:1]:

        layer.trainable = False

    #resnet50.summary()

    return model



def inception():

    model = Sequential()

    model.add(layers.Input(shape=(img_size,img_size,3)))

    model.add(tf.keras.applications.InceptionV3(include_top=False,weights="imagenet"))

    model.add(layers. GlobalAveragePooling2D())#BatchNormalization()

    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(3,activation='softmax'))

    for layer in model.layers[:1]:

        layer.trainable = False

    #inception.summary()

    return model



def densenet():

    model = Sequential()

    model.add(layers.Input(shape=(img_size,img_size,3)))

    model.add(tf.keras.applications.DenseNet121(include_top=False,weights="imagenet"))

    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dense(256,activation='relu'))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(3,activation='softmax'))

    for layer in model.layers[:1]:

        layer.trainable = False

    #inception.summary()

    return model



def vgg():

    model = Sequential()

    model.add(layers.Input(shape=(img_size,img_size,3)))

    model.add(tf.keras.applications.VGG16(include_top=False,weights="imagenet"))

    model.add(layers.Flatten())

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(3,activation='softmax'))

    for layer in model.layers[:1]:

        layer.trainable = False

    #inception.summary()

    return model

models={'mobilenet':mobilenet,'xception':xception,'inception':inception,'densenet':densenet,'vgg':vgg}

paths=['mobilenet.h5','xception.h5','inception.h5','densenet.h5','vgg16.h5']
def history_plot(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(len(acc))



    fig, axes = plt.subplots(1, 2, figsize=(15,5))



    axes[0].plot(epochs, acc, 'r-', label='Training Accuracy')

    axes[0].plot(epochs, val_acc, 'b--', label='Validation Accuracy')

    axes[0].set_title('Training and Validation Accuracy')

    axes[0].legend(loc='best')



    axes[1].plot(epochs, loss, 'r-', label='Training Loss')

    axes[1].plot(epochs, val_loss, 'b--', label='Validation Loss')

    axes[1].set_title('Training and Validation Loss')

    axes[1].legend(loc='best')



    plt.show()
%%time

#hist=[]

print('Training: ')

for m,model_path,tr in zip(models,paths,sub.T):

    model=models[m]()

    train_sub=train[tr]

    #print(train_sub.shape)

    df_train_sub=df_train.values[tr]

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history = model.fit(train_sub,df_train_sub,batch_size=128,epochs=25,validation_split=0.1,callbacks=[checkpoint_cb],verbose=0)

    print('Model: ',m)

    #hist.append(history)

    history_plot(history)

    del model, history

    gc.collect()

    print('')

# del models
labels=[]

for m,model_path in zip(models,paths):

    model = load_model(model_path)

    print('Model: ',m,' Acc: ',model.evaluate(test,df_test,verbose=0)[1])

    y_pred=model.predict(test,verbose=0)

    y_pred=np.argmax(y_pred,axis=1)

    labels.append(y_pred)

    print('')

    del model

    gc.collect()

labels = np.array(labels)

labels=labels.T

y=stats.mode(labels,axis=1)[0]

print('Ensemble: ',accuracy_score(y,np.argmax(df_test.values,axis=1)))