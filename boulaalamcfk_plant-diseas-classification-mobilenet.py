import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
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
# base_model = keras.applications.MobileNet(weights="imagenet",include_top=False,input_shape=(image_size,image_size,3))

# base_model.trainable = False
# inputs = keras.Input(shape=(image_size,image_size,3))
# x = base_model(inputs,training=False)
# x = keras.layers.GlobalAveragePooling2D()(x)
# x = keras.layers.Dropout(0.2)(x)
# x = keras.layers.Dense(len(categories),activation="softmax")(x)
# model = keras.Model(inputs=inputs, outputs=x, name="LeafDisease_MobileNet")
# model.summary()

# optimizer = keras.optimizers.Adam()
# model.compile(optimizer=optimizer,loss=keras.losses.CategoricalCrossentropy(from_logits=True),metrics=[keras.metrics.CategoricalAccuracy()])

# move the model 
%cp -arvf "../input/leaf-cnn-mobelnet/leaf-cnn.h5" ./

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
# define the checkpoint
filepath = "./leaf-cnn.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
callbacks_list = [checkpoint]

# load the model
model = load_model(filepath)
history = model.fit_generator(train_data,
          validation_data=test_data,
          epochs=100,
          steps_per_epoch=150,
          validation_steps=100
         )
print("[INFO] Calculating model accuracy")
scores = model.evaluate(test_data)
print(f"Test Accuracy: {scores[1]}")
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
# predicting an image
import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
directory="../input/new-plant-diseases-dataset/test/test"
files = [os.path.join(directory,p) for p in sorted(os.listdir(directory))]
for i in range(0,10):
    image_path = files[i]
    new_img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(new_img)
    img = np.expand_dims(img, axis=0)
    img = img/255
    prediction = model.predict(img)
    probabilty = prediction.flatten()
    max_prob = probabilty.max()
    index=prediction.argmax(axis=-1)[0]
    class_name = categories[index]
    #ploting image with predicted class name        
    plt.figure(figsize = (4,4))
    plt.imshow(new_img)
    plt.axis('off')
    plt.title(class_name+" "+ str(max_prob)[0:4]+"%")
    plt.show()
        
valid_num = test_data.samples
print("valid_num is:",valid_num)
#Confution Matrix and Classification Report
from sklearn.metrics import classification_report, confusion_matrix
y_pred = model.predict_generator(test_data, valid_num//batch_size+1)
class_dict = test_data.class_indices
li = list(class_dict.keys())
print(li)
y_pred = np.argmax(y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_data.classes, y_pred))
print('Classification Report')
target_names =li ## ['Peach___Bacterial_spot', 'Grape___Black_rot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
print(classification_report(test_data.classes, y_pred, target_names=target_names))
# save the model to disk
print("[INFO] Saving model...")
filepath="leaf-cnn.h5"
model.save(filepath)
