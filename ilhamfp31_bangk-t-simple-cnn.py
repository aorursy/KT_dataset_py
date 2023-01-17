import os

kue_putri_salju_dir = os.path.join('../input/kue-indonesia/train/kue_putri_salju')

kue_klepon_dir = os.path.join('../input/kue-indonesia/train/kue_klepon')



print('total training kue putri salju images:', len(os.listdir(kue_putri_salju_dir)))

print('total training kue klepon images:', len(os.listdir(kue_klepon_dir)))



kue_putri_salju_files = os.listdir(kue_putri_salju_dir)

print(kue_putri_salju_files[:10])



kue_klepon_files = os.listdir(kue_klepon_dir)

print(kue_klepon_files[:10])
%matplotlib inline



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



pic_index = 12



next_kue_putri_salju = [os.path.join(kue_putri_salju_dir, fname) 

                        for fname in kue_putri_salju_files[pic_index-2:pic_index]]

next_kue_klepon = [os.path.join(kue_klepon_dir, fname) 

                   for fname in kue_klepon_files[pic_index-2:pic_index]]





for i, img_path in enumerate(next_kue_putri_salju+next_kue_klepon):

  #print(img_path)

  img = mpimg.imread(img_path)

  plt.imshow(img)

  plt.axis('Off')

  plt.show()
import tensorflow as tf

import keras_preprocessing

from keras_preprocessing import image

from keras_preprocessing.image import ImageDataGenerator



print("Loading training data...\t\t", end='')

TRAINING_DIR = "../input/kue-indonesia/train/"

training_datagen = ImageDataGenerator(

      rescale = 1./255,

	  rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest',)



train_generator = training_datagen.flow_from_directory(

	TRAINING_DIR,

	target_size=(150,150),

	class_mode='categorical',

    shuffle=True,

    seed=42,

    batch_size=64

)



print("Loading validation data...\t\t", end='')

VALIDATION_DIR = "../input/kue-indonesia/validation/"

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = validation_datagen.flow_from_directory(

	VALIDATION_DIR,

	target_size=(150,150),

	class_mode='categorical',

    batch_size=16

)



print("Loading test data...\t\t\t", end='')

TEST_DIR = "../input/kue-indonesia/test/"

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(

	TEST_DIR,

	target_size=(150,150),

	class_mode='categorical',

    batch_size=16

)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping



def callbacks():

    cb = []



    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',  

                                       factor=0.5, patience=1, 

                                       verbose=1, mode='min', 

                                       epsilon=0.0001, min_lr=0,

                                       restore_best_weights=True)

    cb.append(reduceLROnPlat)

    

    log = CSVLogger('log.csv')

    cb.append(log)

    

    es = EarlyStopping(monitor='val_loss', patience=5, verbose=0,

                       mode='min', restore_best_weights=True)

    

    cb.append(es)

    

    return cb
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),



    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),



    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),



    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.3),



    tf.keras.layers.Dense(256, activation='relu'),

    tf.keras.layers.Dense(8, activation='softmax')

])





model.summary()



model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## Calculate training time

import time

start = time.time()
history = model.fit(train_generator, 

                    epochs=200, 

                    validation_data = validation_generator, 

                    verbose = 1,

                    callbacks = callbacks())
print("Time elapsed training: ")

end = time.time()

print("{} second".format(end - start))
model.save("model.h5")
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
results = model.evaluate(test_generator)
print('test loss, test acc:', results)