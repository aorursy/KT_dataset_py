import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import cv2
import os
where_img = pd.read_csv('img-class.csv') #使用這個cvs讀出各資料夾類別
print(where_img.head(20))
from sklearn.model_selection import train_test_split #分割成 train 和 validation兩個資料集
train_pd, valid_pd = train_test_split(where_img, test_size = 0.1,
                                      shuffle = True, random_state = 95)

from sklearn.utils import shuffle
from keras.utils import to_categorical

#手刻generator避免ram爆掉
def data_gen(df,batch_size, reshape_size):
    
    df2 = shuffle(df)
    y = df2.classname
    x = df2.img
    
    out_x = []
    out_y = []
    while True:
        for i in range(len(x)):
            #read img from path
            img = cv2.imread('./train/'+x.iloc[i])
            img = cv2.resize(img,(reshape_size, reshape_size))
            img = img.reshape(reshape_size, reshape_size, 3)
            img = img /255.0
            out_x.append(img)
            out_y.append(y.iloc[i])

            if len(out_x)==batch_size:
                ox = np.array(out_x)
                oy = np.array(to_categorical(out_y,15))
                out_x = []
                out_y = []
                yield ox, oy
cv2.imread('./train/'+'bedroom/image_0013.jpg').shape
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import os

batch_size = 32
num_classes = 15
epochs = 50
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'where_am_I.h5'

width=256
height=256


# build our CNN model
model = Sequential()
model.add(Conv2D(128, (3, 3), padding='same',input_shape=(256,256,3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (5,5), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(512, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))


model.add(GlobalAveragePooling2D()) #不用 Flatten() 以避免 oom
# model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())

# initiate Adam optimizer，用小一點數值
opt = keras.optimizers.Adam(1e-4)

# Let's train the model using Adam
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


print('Using real-time data augmentation.')

train_generator = data_gen(train_pd, 16, 256)

valid_generator = data_gen(valid_pd, 16, 256)


# 用 ModelCheckpoint  存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(model_path, monitor='acc', save_best_only=True, verbose=1)

# earlystop
earlystop = EarlyStopping(monitor='val_loss', patience=6, verbose=1)


# Fit the model on the batches generated 
model_history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=500,
                    epochs=epochs,
                    validation_data=valid_generator,
                    validation_steps=299/16,
                    workers=4,
                    callbacks=[earlystop, checkpoint])

# loading our save model
print("Loading trained model")
model = load_model(model_path)
from PIL import Image # 讀圖片用的套件

x_test = [] # 用來存轉成數字陣列的圖片，也就是你要輸進模型的 feature
ids = []
for image_name in os.listdir("testset"): # os.listdir 可以列出資料夾裡的所有檔案名稱，括號裡放 test1 代表我要看 test1 資料夾裡的所有檔案
    
    img = Image.open("testset/" + image_name) # Image.open 可以開啟圖片檔案，不過他還不是陣列，所以我們還要再做轉換
    img = img.resize((256,256)) # 把圖片統一轉成 256 * 256
    img = np.array(img) # 把剛剛開啟的圖片放到 numpy.array 裡就可以轉成陣列
    if img.shape != (256,256):
        img = img[:,:,0]
        
    img = np.stack((img, img, img), axis=2)
    ids.append(image_name.split(".jpg")[0]) # 把檔名前的 id 抓出來
    x_test.append(img)
x_test = np.array(x_test).reshape(1500,256,256,3) # 模型只吃 numpy.array 型態的資料，所以先把 x_test 轉成 numpy.array
x_test = x_test / 255 # 把所有 pixel 標準化到 0 到 1 之間
labels = model.predict_classes(x_test) # 把 x_test 丟到 predict_classes 就會回傳預測的類別
dataFrame = pd.DataFrame({"id":ids,"class":labels}) # 把 label 和 id 丟到 DataFrame，方便等一下輸出 csv
#d2 = dataFrame.replace({0: 12, 1: 9, 2:10, 3:4, 4:14, 5:2, 6:3, 7:0, 8:5, 9:8, 10:6, 11:7, 12:11, 13:1,14:13}) #取代掉類別，這行不用理他
dataFrame.to_csv("my_submit.csv",index = False) # 輸出 csv 檔，index = False 的話，csv 檔左邊就不會多一排 index
training_loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.plot(training_loss, label="training_loss")
plt.plot(val_loss, label="validation_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Learning Curve")
plt.legend(loc='best')
plt.show()
