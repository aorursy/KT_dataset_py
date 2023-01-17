import numpy as np

import pandas as pd

import keras as k
train = pd.read_json('../input/train.json')

train.inc_angle = train.inc_angle.replace('na',0)
def transform(df):

    images = []

    for i, row in df.iterrows():

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 / band_2

    

        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())

        band_2_norm = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())

        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))

    return np.array(images)
def augment(images):

    image_mirror_lr = []

    image_mirror_ud = []

    for i in range(0,images.shape[0]):

        band_1 = images[i,:,:,0]

        band_2 = images[i,:,:,1]

        band_3 = images[i,:,:,2]

            

        # mirror left-right

        band_1_mirror_lr = np.flip(band_1, 0)

        band_2_mirror_lr = np.flip(band_2, 0)

        band_3_mirror_lr = np.flip(band_3, 0)

        image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr)))

        

        # mirror up-down

        band_1_mirror_ud = np.flip(band_1, 1)

        band_2_mirror_ud = np.flip(band_2, 1)

        band_3_mirror_ud = np.flip(band_3, 1)

        image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud)))

        

    mirrorlr = np.array(image_mirror_lr)

    mirrorud = np.array(image_mirror_ud)

    images = np.concatenate((images, mirrorlr, mirrorud))

    return images
train_X = transform(train)

train_y = np.array(train['is_iceberg'])

train_X = augment(train_X)

train_y = np.concatenate((train_y, train_y, train_y))
model=k.models.Sequential()

model.add(k.layers.convolutional.Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))

model.add(k.layers.convolutional.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(k.layers.Dropout(0.2))



model.add(k.layers.convolutional.Conv2D(128, kernel_size=(3, 3), activation='relu' ))

model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(k.layers.Dropout(0.2))



model.add(k.layers.convolutional.Conv2D(128, kernel_size=(3, 3), activation='relu' ))

model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(k.layers.Dropout(0.2))



model.add(k.layers.convolutional.Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(k.layers.Dropout(0.2))



model.add(k.layers.Flatten())



model.add(k.layers.Dense(512))

model.add(k.layers.Activation('relu'))

model.add(k.layers.Dropout(0.3))



model.add(k.layers.Dense(256))

model.add(k.layers.Activation('relu'))

model.add(k.layers.Dropout(0.3))



model.add(k.layers.Dense(1))

model.add(k.layers.Activation('sigmoid'))



model.compile(loss='binary_crossentropy', optimizer=k.optimizers.Nadam(0.001), metrics=['accuracy'])

#model.summary()
model.fit(train_X, train_y, batch_size=25, epochs=5, verbose=1, validation_split=0.05)
test = pd.read_json('../input/test.json')

test.inc_angle = train.inc_angle.replace('na',0)

test_X = transform(test)

pred_test = model.predict(test_X)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})

submission.to_csv('submission.csv', index=False)