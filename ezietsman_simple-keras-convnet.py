import time



import pandas as pd

import numpy as np



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.optimizers import Adam

from keras.callbacks import TensorBoard



from sklearn.model_selection import train_test_split
def get_images(df):

    '''Create 2-channel 'images'. Return normalised images.'''

    

    im1 = df.band_1.apply(np.array).apply(lambda x: x.reshape(75, 75)).tolist()

    im2 = df.band_2.apply(np.array).apply(lambda x: x.reshape(75, 75)).tolist()

    

    im1 = np.array(im1)

    im2 = np.array(im2)



    images = np.stack([im1, im2], axis=3)

    

    # normalise images.

    im_min = images.min(axis=(0, 1), keepdims=True)

    im_max = images.max(axis=(0, 1), keepdims=True)

    images = (images - im_min) / (im_max - im_min)

    

    return images
def create_model():

    '''Create and return a keras model.'''

    

    model = Sequential()

    

    # input: 75x75 images with 2 channels 

    

    # this applies 16 convolution filters of size 3x3 each.

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(75, 75, 2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))



    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))



    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))

    

    tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=32)



    return model, [tensorboard]
train = pd.read_json('../input/train.json')

X = get_images(train)

y = train.is_iceberg.values



train = None



X, Xt, y, yt = train_test_split(X, y, test_size=0.25)
model, callbacks = create_model()

model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

model.fit(X, y, validation_data=(Xt, yt), batch_size=32, epochs=5, callbacks=callbacks)
model.save('model.h5')
# create a submission



test = pd.read_json('../input/test.json')

X = get_images(test)

# make predictions

predictions = model.predict_proba(X)

submission = pd.DataFrame({'id': test['id'], 'is_iceberg': predictions[:, 0]})

submission.to_csv('submission.csv', index=False)