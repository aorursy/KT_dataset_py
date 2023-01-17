import tensorflow as tf

import tensorflow_hub as hub

assert tf.__version__.startswith('2')



import os

import numpy as np

import matplotlib.pyplot as plt
tf.__version__
_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"



zip_file = tf.keras.utils.get_file(origin=_URL, 

                                   fname="images.tar", 

                                   extract=True)



base_dir = os.path.join(os.path.dirname(zip_file), 'Images')
IMAGE_SIZE = 299

BATCH_SIZE = 64



datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.2)



train_generator = datagen.flow_from_directory(

    base_dir,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE, 

    subset='training')



val_generator = datagen.flow_from_directory(

    base_dir,

    target_size=(IMAGE_SIZE, IMAGE_SIZE),

    batch_size=BATCH_SIZE, 

    subset='validation')
for image_batch, label_batch in train_generator:

    break

image_batch.shape, label_batch.shape
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)



# https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4

feature_extractor_model = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

feature_extractor_layer = hub.KerasLayer(

    feature_extractor_model, input_shape=IMG_SHAPE, trainable=False)
feature_batch = feature_extractor_layer(image_batch)

print(feature_batch.shape)
model = tf.keras.Sequential([

    feature_extractor_layer,

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')

])



model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(), 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
model.summary()
print('Number of trainable variables = {}'.format(len(model.trainable_variables)))
epochs = 5



history = model.fit(train_generator,

                    epochs=epochs, 

                    validation_data=val_generator)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.title('Training and Validation Loss')

plt.xlabel('epoch')

plt.show()
class_names = sorted(train_generator.class_indices.items(), key=lambda pair:pair[1])

class_names = np.array([key.title() for key, value in class_names])

class_names
for image_batch, label_batch in val_generator:

    break

image_batch.shape, label_batch.shape



predicted_batch = model.predict(image_batch)

predicted_id = np.argmax(predicted_batch, axis=-1)

predicted_label_batch = class_names[predicted_id]
label_id = np.argmax(label_batch, axis=-1)
plt.figure(figsize=(15,20))

plt.subplots_adjust(hspace=0.5)

for n in range(30):

    plt.subplot(6,5,n+1)

    plt.imshow(image_batch[n])

    color = "green" if predicted_id[n] == label_id[n] else "red"

    plt.title(predicted_label_batch[n].title(), color=color)

    plt.axis('off')

    _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")