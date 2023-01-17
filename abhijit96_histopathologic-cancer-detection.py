import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout

from tensorflow.keras import regularizers



print(os.listdir("../input"))
train_df = pd.read_csv('../input/train_labels.csv')

tdf_0 = train_df[train_df.label == 0].sample(frac=0.6807)

tdf_1 = train_df[train_df.label == 1]

train_df = pd.concat([tdf_0, tdf_1])



del tdf_0,tdf_1



train_df = train_df.sample(frac=1).reset_index(drop=True)

train_id, test_id, train_label, test_label = train_test_split(train_df.id.values.tolist(), train_df.label.values.tolist(), test_size = 0.002)

batch_size = 25

epochs = 30

steps = int(len(train_id)/batch_size) + 1
def get_image(path, file):

    img = cv2.imread(os.path.join(path,file+'.tif'))

    #img = img[31:63, 31:63]

    img = cv2.resize(img, (64,64))

    img = img/255

    return img



def get_batch():

    global batch_size

    done = 0

    for i in range(0,len(train_id),batch_size):

        batch_imgs = np.array([get_image('../input/train',train_id[j]) for j in range(done, min(len(train_id),done + batch_size))])

        batch_labels = [train_label[j] for j in range(done, min(len(train_id),done + batch_size))]

        done += batch_size

        yield batch_imgs, batch_labels
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same',\

                 activation = 'relu', input_shape = (64,64,3)))

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same',\

                 activation = 'relu'))

model.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same',\

                 activation = 'relu'))

model.add(Flatten())

#model.add(Dropout(0.03125))

#model.add(Dense(10, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(1, activation = 'sigmoid'))



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
test_images = np.array([get_image('../input/train',test_id[j]) for j in range(len(test_id))])

test_labels = [train_label[j] for j in range(len(test_id))]
for i in range(epochs):

    print('Epoch ' + str(i+1) + ' of ' + str(epochs) + ' :')

    model.fit_generator(get_batch(),steps_per_epoch=steps)

    metric = model.evaluate(test_images, test_labels)

    #print('Accuracy for epoch '+str(i+1)+'=> loss: %f   accuracy: %f'%(metric[0], metric[1]))
test_id = os.listdir('../input/test')

def get_test_image(path, file):

    img = cv2.imread(os.path.join(path,file))

    #img = img[31:63, 31:63]

    img = cv2.resize(img, (64,64))

    img = img/255

    return img

def get_test_batch():

    global batch_size

    done = 0

    for i in range(0,len(test_id),batch_size):

        batch_imgs = np.array([get_test_image('../input/test',test_id[j]) for j in range(done, min(len(test_id),done + batch_size))])

        done += batch_size

        yield batch_imgs

preds = []

for images in get_test_batch():

    preds += np.rint(model.predict(images)).tolist()

preds = np.reshape(preds, (len(preds))).tolist()

df = {'id' : list(map(lambda x : x.split('.')[0], test_id)), 'label': preds}

df = pd.DataFrame(df)

df.to_csv('results.csv', index=False)