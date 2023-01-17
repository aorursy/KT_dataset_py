import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import cv2

from tqdm import tqdm, tqdm_notebook

from keras import layers 

from keras import models

import matplotlib.pyplot as plt

from keras import optimizers

train = '../input/train/train/'

test = "../input/test/test/"

train_csv = pd.read_csv('../input/train.csv')
images_tr = []

labels_tr = []

imges = train_csv['id'].values

for img_id in tqdm_notebook(imges):

    images_tr.append(cv2.imread(train + img_id))    

    labels_tr.append(train_csv[train_csv['id'] == img_id]['has_cactus'].values[0])  

images_tr = np.asarray(images_tr)

images_tr = images_tr.astype('float32')

images_tr /= 255

labels_tr = np.asarray(labels_tr)
#model



model = models.Sequential() 



model.add(layers.Conv2D(32, (3, 3), activation='relu',                     

                                    input_shape=(32, 32, 3))) 

model.add(layers.Conv2D(32, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2))) 



 

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Flatten()) 

model.add(layers.Dense(512, activation='relu')) 

model.add(layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy',      

              optimizer=optimizers.RMSprop(lr=1e-4),         

              metrics=['acc'])





hist = model.fit(images_tr, labels_tr,

                validation_split=0.2,

                batch_size=100,

                epochs = 20,

                )
hist.history.keys()


# summarize history for accuracy

plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
x_tst = []

Test_imgs = []

for img_id in tqdm_notebook(os.listdir(test)):

    x_tst.append(cv2.imread(test + img_id))     

    Test_imgs.append(img_id)

x_tst = np.asarray(x_tst)

x_tst = x_tst.astype('float32')

x_tst /= 255
# Prediction

test_predictions = model.predict(x_tst)
test_predictions[1]
sub_df = pd.DataFrame(test_predictions, columns=['has_cactus'])

sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

sub_df['id'] = ''

cols = sub_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

sub_df=sub_df[cols]

for i, img in enumerate(Test_imgs):

    sub_df.set_value(i,'id',img)
sub_df.to_csv('submission.csv',index=False)