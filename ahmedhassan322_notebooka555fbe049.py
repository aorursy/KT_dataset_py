import numpy as np
import pandas as pd
import json
import gc
import matplotlib.pyplot as plt
%matplotlib inline
from glob import glob
import os
train_files = glob("../input/quickdraw-doodle-recognition/train_simplified/*.csv")
rows = 150000
rows = rows - (rows % 340)
cat_size = rows // 340
print(cat_size)
gc.collect()
from PIL import Image, ImageDraw
from dask import bag
def drawStrokes(matrixOfStrokes):
    image = Image.new("RGB", (256,256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in json.loads(matrixOfStrokes):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=5)
    return np.array(image.resize((32,32)))/255.
drawingArray = np.zeros((rows,32,32,3))
categories = pd.Series([None] * rows)
i = 0
for f in train_files:
    for df in pd.read_csv(f, index_col="key_id", chunksize=1000, nrows=cat_size):
        imagebag = bag.from_sequence(df.drawing.values).map(drawStrokes)
        imagebag = np.array(imagebag.compute())
        categories[i:(i + imagebag.shape[0])] = df["word"].replace("\s+", "_", regex=True)
        drawingArray[i:(i + imagebag.shape[0])] = imagebag
        i += imagebag.shape[0]
        print(i)
gc.collect()
from sklearn.model_selection import train_test_split
indecator = pd.get_dummies(categories)
tr_x,tst_x,tr_indecator,tst_indecator = train_test_split(drawingArray
                                                           , indecator
                                                           , test_size=0.2
                                                           ,random_state=25)
del drawingArray,categories
gc.collect()
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Activation

model = Sequential()
model.add(Conv2D(64, kernel_size=(4,4), strides=1, input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(4,4), strides=1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(340))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(tr_x, tr_indecator,batch_size=200,epochs=30
          ,validation_data=(tst_x,tst_indecator))

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#del tr_x,tst_x,tr_indecator,tst_indecator
gc.collect()
test = pd.read_csv('../input/quickdraw-doodle-recognition/test_simplified.csv', index_col="key_id" ,nrows=100)
ids = test.index
imagebag = bag.from_sequence(test.drawing.values).map(drawStrokes)
test_simplified = np.array(imagebag.compute())
test_simplified = test_simplified.reshape(len(test_simplified), 32, 32, 3)
del imagebag
gc.collect()
prediction = model.predict(test_simplified)
indexOfBigProbability = (-prediction).argsort()[:,:3]
gc.collect()
import matplotlib.pyplot as plt
%matplotlib inline
import ast
import warnings
warnings.filterwarnings('ignore')

raw_images = [ast.literal_eval(lst) for lst in test.loc[test.iloc[:40].index, 'drawing'].values]
j=0
for index, raw_drawing in enumerate(raw_images):
    plt.figure(figsize=(3,3))
    for x,y in raw_drawing:
        title_obj=plt.title(indecator.columns[indexOfBigProbability][j][0]
                  +"  "
                 +indecator.columns[indexOfBigProbability][j][1]
                  +"  "
                 +indecator.columns[indexOfBigProbability][j][2], fontsize=22)
        plt.setp(title_obj, color='green')
        plt.subplot(1, 1, 1)
        plt.plot(x,y)
        plt.axis('off')
    plt.gca().invert_yaxis()
    j+=1
gc.collect()
import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Activation
from keras.applications.vgg16 import VGG16

vgg16 = keras.applications.vgg16.VGG16(weights='imagenet',classes=340,include_top=False,input_shape=(32,32,3))
vgg16.summary()
m = Sequential()
for layer in vgg16.layers:
    m.add(layer)
for layer in m.layers:
    layer.trainable = False
m.add(Flatten())
m.add(Dense(4096,activation='relu'))
m.add(Dense(4096,activation='relu'))
m.add(Dense(340,activation='softmax'))
m.summary()
m.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history1 = m.fit(tr_x, tr_indecator,batch_size=1024,epochs=23,validation_data=(tst_x,tst_indecator))

print(history1.history.keys())
# summarize history for accuracy
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
pred = m.predict(test_simplified)
ind = (-pred).argsort()[:,:3]
gc.collect()
import matplotlib.pyplot as plt
%matplotlib inline
import ast
import warnings
warnings.filterwarnings('ignore')

raw_images = [ast.literal_eval(lst) for lst in test.loc[test.iloc[:80].index, 'drawing'].values]
j=0
for index, raw_drawing in enumerate(raw_images):
    plt.figure(figsize=(3,3))
    for x,y in raw_drawing:
        title_obj=plt.title(indecator.columns[ind][j][0]
                  +"  "
                 +indecator.columns[ind][j][1]
                  +"  "
                 +indecator.columns[ind][j][2], fontsize=22)
        plt.setp(title_obj, color='green')
        plt.subplot(1, 1, 1)
        plt.plot(x,y)
        plt.axis('off')
    plt.gca().invert_yaxis()
    j+=1
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('CNN vs VGG16 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['trainCNN', 'testCNN','trainVGG16','testVGG16'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('CNN vs VGG16 loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['trainCNN', 'testCNN','trainVGG16','testVGG16'], loc='upper left')
plt.show()