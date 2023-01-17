import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sklearn.metrics
import pandas as pd
base_dir = "../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
image_size = 224
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                            shear_range = 0.2,
                                                            zoom_range = 0.2,
                                                            width_shift_range = 0.2,
                                                            height_shift_range = 0.2,
                                                            fill_mode="nearest")
batch_size = 32
train_data = train_datagen.flow_from_directory(os.path.join(base_dir,"train"),
                                               target_size=(image_size,image_size),
                                               batch_size=batch_size,
                                               class_mode="categorical"                                               
                                              )
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)
test_data = test_datagen.flow_from_directory(os.path.join(base_dir,"valid"),
                                               target_size=(image_size,image_size),
                                               batch_size=batch_size,
                                               class_mode="categorical"                                               
                                              )
categories = list(train_data.class_indices.keys())
print(categories)
train_data.image_shape
base_model = keras.applications.MobileNet(weights="imagenet",include_top=False,input_shape=(image_size,image_size,3))

base_model.trainable = False
inputs = keras.Input(shape=(image_size,image_size,3))
x = base_model(inputs,training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.Dense(len(categories),activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=x, name="LeafDisease_MobileNet")
model.summary()
optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer,loss=keras.losses.CategoricalCrossentropy(from_logits=True),metrics=[keras.metrics.CategoricalAccuracy()])
history = model.fit_generator(train_data,
          validation_data=test_data,
          epochs=25,
          steps_per_epoch=150,
          validation_steps=100
         )
model.evaluate(test_data)
model.save('leaf-cnn.h5')
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,loss,c="red",label="Training")
plt.plot(epochs,val_loss,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
# reset the test_data to start iterating over dataset from scratch
test_data.reset()
# start to predict
pred = model.predict(test_data)
# use the confusion_matrix function provided by tensorflow to generate confusion matrix
con_mat = tf.math.confusion_matrix(labels=test_data.classes, predictions=np.argmax(pred, axis=1)).numpy()

# normalize the confusion matrix
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

# convert the nomalized confusion matrix for better view
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = test_data.class_indices.keys(), 
                     columns = test_data.class_indices.keys())

# show the nomalized confusion matrix
con_mat_df
# convert the original confusion matrix for better view (using the case numbers)
con_mat_df_explain = pd.DataFrame(con_mat,
                     index = test_data.class_indices.keys(), 
                     columns = test_data.class_indices.keys())

# show the unnomalized confusion matrix
con_mat_df_explain
# generate the clasification report by using the classification_report of sklearn package
report = sklearn.metrics.classification_report(test_data.classes, np.argmax(pred, axis=1), target_names=test_data.class_indices.keys())

# print the report
print(report)