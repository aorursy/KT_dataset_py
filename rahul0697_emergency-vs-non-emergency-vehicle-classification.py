import pandas as pd
import os
Data_dir = "/kaggle/input"
os.listdir(Data_dir)
train_labels = pd.read_csv('/kaggle/input/emergencyvsnonemergency/train.csv')
test_labels = pd.read_csv('/kaggle/input/hp-2020/jh_2020/test.csv')
split = len(train_labels)
X_train_dir = os.path.join(Data_dir , "images") 
X_test_dir = os.path.join(Data_dir , "images")
from sklearn.model_selection import train_test_split
X_train , X_valid = train_test_split(train_labels, test_size=0.20, random_state=42)
X_test = test_labels
print(X_train.shape)
print(X_valid.shape)
print(X_test.shape)
import tensorflow as tf
def preprocess(path):
    img = tf.keras.preprocessing.image.load_img(path, color_mode='rgb')
    return img
train_img = X_train['image_names'].tolist()
valid_img = X_valid['image_names'].tolist()
test_img = X_test['image_names'].tolist()
train = []
valid = []
test = []
for i in range(len(train_img)):
    train.append(preprocess(Data_dir + "/emergencyvsnonemergency/images/" + train_img[i]))
    
for i in range(len(valid_img)):
    valid.append(preprocess(Data_dir + "/emergencyvsnonemergency/images/" + valid_img[i]))

for i in range(len(test_labels)):
    test.append(preprocess(Data_dir + "/emergencyvsnonemergency/images/" + test_img[i]))


x_train = []
for i  in range(len(train)):
    x_train.append(tf.keras.preprocessing.image.img_to_array(train[i]))
x_valid = []
for i in range(len(valid)):
    x_valid.append(tf.keras.preprocessing.image.img_to_array(valid[i]))
x_test = []
for i in range(len(test)):
    x_test.append(tf.keras.preprocessing.image.img_to_array(test[i]))

import numpy as np
x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
y_train = np.array(X_train['emergency_or_not'])
y_valid = np.array(X_valid['emergency_or_not'])
print(y_train.shape)
print(y_valid.shape)
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
# from tensorflow.keras.applications import ResNet101 
pretrained_model = MobileNetV2(weights = 'imagenet', include_top=False, input_shape=(224,224,3))
#                                ,pooling='max')
# last_layer = pretrained_model.get_layer[]
for layer in pretrained_model.layers:
    layer.trainable = False
# pretrained_model.summary()
# last_layer = pretrained_model.output
model = keras.Sequential([
    pretrained_model,
#     keras.layers.GlobalAveragePooling2D(),
#     keras.layers.GlobalMaxPool2D(),
#     keras.layers.Flatten(),
#     keras.layers.Dropout(0.25),
    keras.layers.GlobalAveragePooling2D(),
#     keras.layers.Dropout(0.50),
    keras.layers.Dense(128, activation='relu'),
#     keras.layers.BatchNormalization(),
#     keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
train_datagen =  keras.preprocessing.image.ImageDataGenerator(
rescale = 1./255,
rotation_range = 40,
width_shift_range = 0.2,
height_shift_range =0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode= 'nearest'
)

valid_datagen =  keras.preprocessing.image.ImageDataGenerator(
rescale = 1./255,
rotation_range = 40,
width_shift_range = 0.2,
height_shift_range =0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode= 'nearest'
)


train_dataGenerator = train_datagen.flow(x_train, y_train, batch_size=64)
valid_dataGenerator = valid_datagen.flow(x_valid, y_valid, batch_size=64)
model.compile(optimizer= keras.optimizers.Adam(lr=1e-3), loss = 'binary_crossentropy', metrics=['accuracy'])
# export_path = os.path.join(os.getcwd(), 'model', '2016')
checkpoint_filepath = os.path.join(os.getcwd() , 'checkpoint')
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_filepath,
#     save_weights_only=True,
#     monitor='val_acc',
#     mode='max',
#     save_best_only=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath= checkpoint_filepath, save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
history = model.fit(train_dataGenerator, validation_data=valid_dataGenerator,
                    epochs=1, callbacks=[checkpoint_cb])
model = keras.models.load_model(checkpoint_filepath)
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(len(acc))


plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')


plt.figure()


plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()


plt.show()
# model.load_weights(checkpoint_filepath)
x_test = x_test/255.
# x_test[0]
classification = model.predict_classes(x_test)
labels = classification.tolist()
label = []
for i in range(len(labels)):
    label.append(labels[i][0])
sample_submission = pd.read_csv("/kaggle/input/hp-2020/jh_2020/sample_submission.csv")
sample_submission.emergency_or_not = label
sample_submission.head()
sample_submission.to_csv("/kaggle/working/sample_submission8.csv", index=False)
