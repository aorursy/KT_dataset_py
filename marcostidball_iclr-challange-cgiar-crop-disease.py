import os, shutil

import pandas as pd

import numpy as np

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras import models

from keras import layers

from keras import optimizers

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight

import matplotlib.pyplot as plt
leaf_rust_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/leaf_rust/'

healthy_wheat_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/healthy_wheat/'

stem_rust_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/stem_rust/'



sample_img = load_img('/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/train/train/leaf_rust/SS98C9.jpg')

plt.imshow(sample_img)
#shutil.rmtree('/kaggle/working/allimgs')
categories_dir = [leaf_rust_dir, healthy_wheat_dir, stem_rust_dir]



base_dir = '/kaggle/working/allimgs'

os.mkdir(base_dir)



for category in categories_dir:

    images = os.listdir(category)

    ide = []

    lr = []

    hw = []

    sr = []

    

    for img in images:

        img_name = img.split('.')[0]

        img_format = img.split('.')[1]



        im = Image.open(os.path.join(category, img))

        rgb_im = im.convert('RGB')

        img = img_name + '.jpeg'

        rgb_im.save(os.path.join(base_dir, img), 'JPEG', quality=90)

        

        if category == leaf_rust_dir:

            ide.append(img)

            lr.append(1)

            hw.append(0)

            sr.append(0)

            df_lr = pd.DataFrame({'ID': ide, 'leaf_rust':lr, 'stem_rust':sr, 'healthy_wheat':hw })

        

        if category == healthy_wheat_dir:

            ide.append(img)

            lr.append(0)

            hw.append(1)

            sr.append(0)

            df_hw = pd.DataFrame({'ID': ide, 'leaf_rust':lr, 'stem_rust':sr, 'healthy_wheat':hw })



        if category == stem_rust_dir:

            ide.append(img)

            lr.append(0)

            hw.append(0)

            sr.append(1)

            df_sr = pd.DataFrame({'ID': ide, 'leaf_rust':lr, 'stem_rust':sr, 'healthy_wheat':hw })
frames = [df_lr, df_hw, df_sr]

df = pd.concat(frames)

df.reset_index(drop=True, inplace=True)

df.sum()
from keras.applications.xception import Xception



SIZE = 512

BATCH_SIZE = 10



conv_base = Xception(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))



model = models.Sequential()

model.add(conv_base)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(3, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()
train_datagen = ImageDataGenerator(rescale=1./255,

                                   zoom_range=0.3,

                                  rotation_range=40,

                                  horizontal_flip=True,

                                  fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)
columns = ['leaf_rust', 'stem_rust', 'healthy_wheat']



train_df, validation_df = train_test_split(df, test_size=0.1, random_state=42)



weight_df = train_df.drop('ID', axis=1)

y = weight_df.idxmax(axis=1)



class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

class_weights = dict(enumerate(class_weights))
train_size = train_df.shape[0]

validation_size = validation_df.shape[0]



train_generator = train_datagen.flow_from_dataframe(train_df,

                                                   '/kaggle/working/allimgs',

                                                   x_col='ID',

                                                   y_col=columns,

                                                   batch_size=BATCH_SIZE,

                                                   target_size=(SIZE,SIZE),

                                                   class_mode='raw',

                                                   seed=42)



validation_generator = train_datagen.flow_from_dataframe(validation_df,

                                                         '/kaggle/working/allimgs',

                                                         x_col='ID',

                                                         y_col=columns,

                                                         batch_size=BATCH_SIZE,

                                                         target_size=(SIZE,SIZE),

                                                         class_mode='raw',

                                                         seed=42)

del df

del weight_df

del y

del df_lr

del df_hw

del df_sr
history = model.fit_generator(train_generator,

                             steps_per_epoch=train_size//BATCH_SIZE,

                             epochs=30,

                             validation_data=validation_generator,

                             validation_steps=validation_size//BATCH_SIZE,

                             class_weight=class_weights)



model.save('CGIAR-crop_disease.h5')
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'b', label='training acc')

plt.plot(epochs, val_acc, 'r', label='validation acc')

plt.title('accuracy')

plt.legend()



plt.figure()

plt.plot(epochs, loss, 'b', label='training loss')

plt.plot(epochs, val_loss, 'r', label='validation loss')

plt.title('loss')

plt.legend()



plt.show()
#shutil.rmtree('/kaggle/working/testings')
base_test_dir = '/kaggle/input/cgiar-computer-vision-for-crop-disease/ICLR/test/test/'

test_dir = '/kaggle/working/testings' 

os.mkdir(test_dir)

filenames = os.listdir(base_test_dir)



ide_test = []

for img in filenames:

    img_name = img.split('.')[0]

    img_format = img.split('.')[1]



    im = Image.open(os.path.join(base_test_dir, img))

    rgb_im = im.convert('RGB')

    img = img_name + '.jpeg'

    rgb_im.save(os.path.join(test_dir, img), 'JPEG', quality=90)

    ide_test.append(img)

    

test_df = pd.DataFrame({'ID': ide_test})

test_size = test_df.shape[0]



test_generator = train_datagen.flow_from_dataframe(test_df,

                                                   test_dir,

                                                   x_col='ID',

                                                   y_col=None,

                                                   batch_size=BATCH_SIZE,

                                                   target_size=(SIZE,SIZE),

                                                   class_mode=None,

                                                   seed=42)



test_df
test_generator.reset()

predictions = model.predict_generator(test_generator, steps=test_size//BATCH_SIZE)



predictions
results = pd.DataFrame(predictions, columns=columns)



results["ID"] = test_generator.filenames



ordered_cols = ["ID"] + columns

results = results[ordered_cols]



results.to_csv("results.csv", index=False)