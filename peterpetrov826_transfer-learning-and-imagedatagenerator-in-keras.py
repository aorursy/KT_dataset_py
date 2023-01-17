import os



def list_files(startpath):

    for root, dirs, files in os.walk(startpath):

        level = root.replace(startpath, '').count(os.sep)

        indent = ' ' * 4 * (level)

        print('{}{}/'.format(indent, os.path.basename(root)))

        subindent = ' ' * 4 * (level + 1)

        for f in files[:3]:

            print('{}{}'.format(subindent, f))
list_files("../input")
from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_width, image_height = 150,150



train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest'

)



test_datagen = ImageDataGenerator( rescale = 1./255. )



train_generator = train_datagen.flow_from_directory(

                    "../input/seg_train/seg_train/",

                    batch_size=128,

                    class_mode='categorical',

                    target_size=(image_width, image_height)

)     



test_generator =  test_datagen.flow_from_directory(

                    "../input/seg_test/seg_test/",

                    batch_size=128, 

                    class_mode='categorical',

                    target_size=(image_width, image_height)

)
from tensorflow.keras.applications.vgg16 import VGG16



vgg16 = VGG16(

    input_shape=(image_width,image_height,3),

    include_top=False,

    weights="imagenet"

)



for layer in vgg16.layers:

    layer.trainable=False

    

vgg16.summary()
import tensorflow as tf

from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization



model = tf.keras.models.Sequential([

    vgg16,

    

    Flatten(),

    Dense(512,activation="relu"),

    BatchNormalization(),

    Dropout(0.5),

    

    Dense(64,activation="relu"),

    BatchNormalization(),

    Dropout(0.5),

    Dense(6, activation='softmax')

])



model.summary()
model.compile(

            optimizer="adam",

            loss='categorical_crossentropy',

            metrics = ['acc']

)
history = model.fit_generator(

            train_generator,

            validation_data=test_generator,

            epochs=5

)
import matplotlib.pyplot as plt



acc      = history.history['acc']

val_acc  = history.history['val_acc']

loss     = history.history['loss']

val_loss = history.history['val_loss']



epochs   = range(len(acc))





# Plot training and validation accuracy per epoch

plt.plot(epochs,acc)

plt.plot(epochs,val_acc)

plt.title('Training and validation accuracy')

plt.figure()



# Plot training and validation loss per epoch

plt.plot(epochs,loss)

plt.plot(epochs,val_loss)

plt.title('Training and validation loss')
model.save("model_using_vgg16.h5")
predict_datagen = ImageDataGenerator( rescale = 1./255. )



predict_generator =  predict_datagen.flow_from_directory(

                    "../input/seg_pred/",

                    batch_size=128, 

                    class_mode=None,

                    shuffle=False,

                    target_size=(image_width, image_height)

)
pred = model.predict_generator(predict_generator,verbose=1)
import numpy as np

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)

print(labels)
labels = dict((v,k) for k,v in labels.items())

print(labels)
predictions = [labels[k] for k in predicted_class_indices]

print(predictions[:10])