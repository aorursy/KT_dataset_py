import os

import zipfile

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.optimizers import RMSprop

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import load_img
for input_zip in ['../input/dogs-vs-cats/train.zip','../input/dogs-vs-cats/test1.zip']:

    ex_zip = zipfile.ZipFile(input_zip, 'r')

    ex_zip.extractall('.')

    ex_zip.close()
print(os.listdir('./train'))
filenames = os.listdir('./train')

categories = []

for filename in filenames:

    category = filename.split('.')[0]

    if category == 'cat':

        categories.append('cat')

    else:

        categories.append('dog')



df = pd.DataFrame({

    'filename': filenames,

    'category': categories

})
nrows = 5

ncols = 5

img_count = nrows+ncols



df_cats = df.loc[df['category'] == 'cat'].head(img_count)

df_dogs = df.loc[df['category'] == 'dog'].head(img_count)



fig = plt.gcf()

fig.set_size_inches(ncols*4, nrows*4)



next_cat_pix = [os.path.join('./train/', fname) 

                for fname in df_cats.loc[:,'filename'] 

               ]



next_dog_pix = [os.path.join('./train/', fname) 

                for fname in df_dogs.loc[:,'filename']

               ]



for i, img_path in enumerate(next_cat_pix+next_dog_pix):

    sp = plt.subplot(nrows, ncols, i + 1)

    img = mpimg.imread(img_path)

    plt.imshow(img)



print('First '+str(2*img_count)+' images of cats and dogs')

plt.show()
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=50)

train_df = train_df.reset_index(drop=True)

validate_df = validate_df.reset_index(drop=True)



print('Train: '+str(train_df.shape[0]))

print('Validate: '+str(validate_df.shape[0]))
train_datagen = ImageDataGenerator(

    rotation_range=15,

    rescale=1./255,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    width_shift_range=0.1,

    height_shift_range=0.1

)



train_generator = train_datagen.flow_from_dataframe(

    train_df, 

    './train/', 

    x_col='filename',

    y_col='category',

    target_size=(150,150),

    class_mode='categorical',

    batch_size=20

)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(

    validate_df, 

    './train/', 

    x_col='filename',

    y_col='category',

    target_size=(150,150),

    class_mode='categorical',

    batch_size=20

)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(), 

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(2, activation='softmax')  

])
model.summary()
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.99):

            print("\nReached 99% accuracy so cancelling training!")

            self.model.stop_training = True



callbacks = myCallback()
opt = RMSprop(lr=0.001)



model.compile(optimizer=opt,

              loss='binary_crossentropy',

              metrics = ['accuracy'])
history = model.fit(train_generator,

                    validation_data=validation_generator,

                    steps_per_epoch=20000//20,

                    epochs=10,

                    validation_steps=5000//20,

                    verbose=1,

                    callbacks=[callbacks]

                   )
model.save_weights("model.h5")
acc      = history.history[     'accuracy' ]

val_acc  = history.history[ 'val_accuracy' ]

loss     = history.history[    'loss' ]

val_loss = history.history['val_loss' ]



epochs   = range(len(acc))



plt.plot  ( epochs,     acc )

plt.plot  ( epochs, val_acc )

plt.title ('Training and validation accuracy')

plt.figure()



plt.plot  ( epochs,     loss )

plt.plot  ( epochs, val_loss )

plt.title ('Training and validation loss'   )
test_filenames = os.listdir('./test1')

test_df = pd.DataFrame({'filename': test_filenames})
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(

    test_df, 

    './test1', 

    x_col='filename',

    y_col=None,

    class_mode=None,

    target_size=(150,150),

    batch_size=20,

    shuffle=False

)
predict = model.predict_generator(test_generator, steps=np.ceil(12500/20))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

test_df
submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)