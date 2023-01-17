# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from matplotlib import pyplot as plt

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

from keras.layers import Conv2D,MaxPooling2D,GlobalMaxPool2D,Dropout,Dense,Flatten,BatchNormalization

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import numpy as np

from keras.models import Sequential

from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

from sklearn.metrics import roc_auc_score, roc_curve, f1_score

from sklearn.utils import class_weight
output_dir = '../input/aerial-cactus-identification/model_output/CNN'

seed = 7

np.random.seed(seed)
train_df = pd.read_csv("../input/train.csv")

train_df.head()
class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train_df['has_cactus']),

                                                 train_df['has_cactus'])

print(class_weights)
train_image = []



for id in tqdm(range(len(train_df))):

    img = image.load_img('../input/train/train/'+train_df['id'][id],target_size=(32,32)) 

#     plt.imshow(img)

#     break

    img = image.img_to_array(img)

    img = img/255

    train_image.append(img)

X = np.array(train_image)
X.shape
plt.imshow(X[1])
y = np.array(train_df.drop(['id'],axis=1))

y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
img_gen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,zoom_range=0.1,rotation_range=40,brightness_range=(0.5,1.0),

                             height_shift_range=0.2,width_shift_range=0.2)



test_datagen = ImageDataGenerator()

validation_generator = test_datagen.flow(X_test, y_test)
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu", input_shape=(32,32,3)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()
callbacks = [

ModelCheckpoint(filepath="weights.best.hdf5",monitor='val_acc',save_best_only=True, mode='max'),

EarlyStopping(monitor="val_loss",mode='auto',patience=20,restore_best_weights=True),

ReduceLROnPlateau(monitor='val_loss',mode='auto',patience=3,min_lr=0.0001)

]

# if not os.path.exists(output_dir):

#     os.makedirs(output_dir)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test), batch_size=32,shuffle=True,callbacks=callbacks)

# model.fit_generator(img_gen.flow(X_train, y_train), epochs = 100,steps_per_epoch=5000,validation_data=validation_generator,validation_steps=109,shuffle=True,class_weight=class_weights,callbacks=callbacks,use_multiprocessing=True)
pred = {}

def predictions(imagepath,imagename):

#     print(imagepath,imagename)

    img = image.load_img(imagepath,target_size=(32,32,3))

#     plt.imshow(img)

    img = image.img_to_array(img)

    proba = model.predict(img.reshape(1,32,32,3))   

    pred.update( {imagename : (int(proba[0][0]))} )  

#     print(imagename,np.argmax(proba)+1)

#     print(int(proba[0][0]))
model.load_weights("weights.best.hdf5")
y_hat = model.predict_proba(X_test)

get_auc = roc_auc_score(y_test,y_hat)*100.0

print(get_auc)
files = os.listdir("../input/test/test")

for file in tqdm(files):

    predictions("../input/test/test/"+file,file)

    

pred_df = pd.DataFrame(list(pred.items()), columns=['id', 'has_cactus'])

pred_df.shape,pred_df.head()
pred_df.to_csv(r'Submission.csv',index=False)