import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import numpy as np
tf.__version__
image_n = tf.keras.preprocessing.image.load_img(r'../input/covid-data-gradient-crescent/all/train/normal/NORMAL2-IM-1293-0001.jpeg', target_size=(550,550))
image_n
image_c = tf.keras.preprocessing.image.load_img(r'../input/covid-data-gradient-crescent/all/train/covid/1312A392-67A3-4EBF-9319-810CF6DA5EF6.jpeg', target_size=(550,550))
image_c
train_imagenerator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input, rotation_range = 50, width_shift_range = 0.2,
                                     height_shift_range = 0.2,
                                     zoom_range = 0.1,
                                     horizontal_flip = True,
                                     vertical_flip = True)
train_generator = train_imagenerator.flow_from_directory('../input/covid-data-gradient-crescent/all/train', target_size = (550, 550), batch_size=16,
                                                         class_mode = 'categorical', shuffle = True)
step_size_train = train_generator.n // train_generator.batch_size
step_size_train
test_imagenerator = ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_generator = test_imagenerator.flow_from_directory('../input/covid-data-gradient-crescent/all/test',target_size=(550,550),batch_size=1,
                                                       class_mode = 'categorical',shuffle = False)
step_size_test = test_generator.n // test_generator.batch_size
step_size_test
#loading the model:
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
base_model.summary()
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x) # to do the Pooling is necessary the last layer

#the dense layers:
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
preds = tf.keras.layers.Dense(4, activation='softmax')(x)
model = tf.keras.Model(inputs = base_model.input, outputs = preds)
model.summary()
for i, layer in enumerate(model.layers):
    print(i, layer.name)
for layer in model.layers[:175]:
    layer.trainable = False
for layer in model.layers[175:]:
    layer.trainable = True
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              epochs=25,
                              steps_per_epoch=step_size_train,
                              validation_data = test_generator,
                              validation_steps=step_size_test)
np.mean(history.history['val_accuracy'])
np.std(history.history['val_accuracy'])
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend();
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend();
filenames = test_generator.filenames
filenames
predictions = model.predict_generator(test_generator, steps = len(filenames))
predictions
predictions2 = []

for i in range(len(predictions)):
    predictions2.append(np.argmax(predictions[i]))
predictions2
test_generator.class_indices
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(predictions2, test_generator.classes)
cm = confusion_matrix(predictions2, test_generator.classes)
cm
sns.heatmap(cm, annot=True);
image = tf.keras.preprocessing.image.load_img(r'../input/covid-data-gradient-crescent/all/test/covid/ryct.2020200034.fig5-day0.jpeg', target_size=(550,550))
plt.imshow(image);
image = tf.keras.preprocessing.image.img_to_array(image)
np.shape(image)
image = np.expand_dims(image, axis = 0)
np.shape(image)
image = tf.keras.applications.resnet50.preprocess_input(image)
predictions = model.predict(image)
print(predictions)
prediction = list(train_generator.class_indices)[np.argmax(predictions[0])]
prediction
image2 = tf.keras.preprocessing.image.load_img(r'../input/covid-data-gradient-crescent/all/test/normal/NORMAL2-IM-1406-0001.jpeg', target_size=(550,550))
plt.imshow(image2);
image2 = tf.keras.preprocessing.image.img_to_array(image2)
np.shape(image2)
image2 = np.expand_dims(image2, axis = 0)
np.shape(image2)
image2 = tf.keras.applications.resnet50.preprocess_input(image2)
predictions2 = model.predict(image2)
print(predictions2)
prediction2 = list(train_generator.class_indices)[np.argmax(predictions2[0])]
prediction2