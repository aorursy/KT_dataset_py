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
from tensorflow.keras import layers, optimizers, models, callbacks, applications

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from shutil import copy2
train_dir = '../input/train/train/'

test_dir = '../input/test/test/'



df = pd.read_csv('../input/train.csv')

df.head()
labels = df['has_cactus'].tolist()

img_paths = df['id'].tolist()
import cv2



indices = np.random.randint(0, 17500, size = 16)

fig = plt.figure(figsize = (10, 10))



for i in range(16):

    image = cv2.imread('../input/train/train/{}'.format(img_paths[indices[i]]))

    plot = fig.add_subplot(4, 4, i + 1)

    title = 'No' if labels[indices[i]] == 0 else 'Yes'

    plot.set_title(title)

    plot.imshow(image)

    

plt.show()
def build_model():

    

    model_input = layers.Input(shape = [32, 32, 3])

    X = applications.densenet.DenseNet121(weights = 'imagenet', include_top = False, classes = 1)(model_input)

    X = layers.Flatten()(X)

    model_output = layers.Dense(1, activation = 'sigmoid')(X)

    

    return models.Model(model_input, model_output)

    

model = build_model()

model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr = 0.0001), metrics = ['accuracy'])

model.summary()

def split_dataframe(csv_file):

    df = pd.read_csv(csv_file)

    

    train_df = df.iloc[:14000, :]

    train_df['has_cactus'] = train_df['has_cactus'].apply(str)

    val_df = df.iloc[14000:, :]

    val_df['has_cactus'] = val_df['has_cactus'].apply(str)

    

    return train_df, val_df



train_df, val_df = split_dataframe('../input/train.csv')
from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_df, val_df = split_dataframe('../input/train.csv')



train_gen = ImageDataGenerator(

            rescale = 1./255,

            validation_split = 0.15,

            shear_range = 0.2,

            zoom_range = 0.2,

            horizontal_flip = True,

            vertical_flip = True,

            rotation_range = 10

        )



train_gen = train_gen.flow_from_dataframe(

            dataframe = train_df,

            directory = train_dir,

            x_col = 'id',

            y_col = 'has_cactus',

            target_size = (32, 32),

            color_mode = 'rgb',

            class_mode = 'binary',

            batch_size = 128,

            shuffle = True,

        )



val_gen = ImageDataGenerator(rescale = 1./255, validation_split = 0.15)



val_gen = val_gen.flow_from_dataframe(

            dataframe = val_df,

            directory = train_dir,

            x_col = 'id',

            y_col = 'has_cactus',

            target_size = (32, 32),

            color_mode = 'rgb',

            class_mode = 'binary',

            batch_size = 128,

            shuffle = True,

        )
checkpoint = callbacks.ModelCheckpoint(

            'model.h5',

            monitor = 'val_loss',

            verbose = 0,

            save_best_only = True,

            save_weights_only = False,

            mode = 'auto'

        )



regulate_lr = callbacks.ReduceLROnPlateau(monitor = 'val_loss', min_lr = 1e-5, patience = 3)



model.fit_generator(

    train_gen,

    steps_per_epoch = 14000 // 128,

    validation_data = val_gen,

    validation_steps = 14000 // 128,

    epochs = 60,

    callbacks = [checkpoint, regulate_lr]

)

# model = models.load_model('./models/' + 'model12.h5')

model = models.load_model('model.h5')

model.evaluate_generator(val_gen, verbose = 1)

img_paths = os.listdir('../input/test/test/')

final_model = model



indices = np.random.randint(0, 3500, size = 16)

fig = plt.figure(figsize = (10, 10))



for i in range(16):

    image = cv2.imread('../input/test/test/{}'.format(img_paths[indices[i]]))

    plot = fig.add_subplot(4, 4, i + 1)

    label = final_model.predict(image.reshape((1, 32, 32, 3)))

    title = 'No' if labels[indices[i]] == 0 else 'Yes'

    plot.set_title(title)

    plot.imshow(image)

    

plt.show()

from tqdm import tqdm



submit = pd.read_csv('../input/sample_submission.csv')

test = []



for image in tqdm(submit['id']):

    test.append(cv2.imread('../input/test/test/' + image))



test = np.array(test)

print(test[0].shape)



# print(submit['id'])

test = test / 255

preds = final_model.predict(test, verbose = 1)



submit['has_cactus'] = preds

submit.head(10)



submit.to_csv('sample_submission.csv', index = False)