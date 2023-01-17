import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from collections import Counter
path = os.path.join("../input/")
train_path = os.path.join(path,"food_c101_n10099_r64x64x3.h5")
test_path = os.path.join(path,"food_test_c101_n1000_r64x64x3.h5")
food_train = h5py.File(train_path,"r")
food_test = h5py.File(test_path,"r")
food_train.keys()
f_train_images = np.array(food_train.get("images"))
f_train_cat = np.array(food_train.get("category"))
f_train_cat_names = np.array([elmt.decode() for elmt in food_train.get("category_names")])
f_test_images = np.array(food_test.get("images"))
f_test_cat = np.array(food_test.get("category"))
f_test_cat_names = np.array([elmt.decode() for elmt in food_test.get("category_names")])
plt.figure()
plt.imshow(f_train_images[0])
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.xlabel(f_train_cat_names[np.argmax(f_train_cat[0])])
plt.show()
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(f_train_images[i])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(f_train_cat_names[np.argmax(f_train_cat[i])])
plt.show()
#quantities
quantities = dict(sorted(Counter([f_train_cat_names[i][0] for i in f_train_cat]).items(), key=lambda item: item[1],
                        reverse=True))
model = keras.Sequential([
    keras.layers.Conv2D(16, (3,3), padding='same', activation='relu', input_shape=f_train_images.shape[1:]),
    keras.layers.MaxPooling2D(),
    #keras.layers.Dropout(0.25),
    #keras.layers.Conv2D(32, (3,3), padding='same', activation='relu'),
    #keras.layers.MaxPooling2D(),
    #keras.layers.Dropout(0.25),
    #keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
    #keras.layers.MaxPooling2D(),
    #keras.layers.Dropout(0.25),
    #keras.layers.Conv2D(128, (3,3), padding='same', activation='relu'),
    #keras.layers.MaxPooling2D(),
    #keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(f_train_cat.shape[1], activation=tf.nn.softmax)
])

model.summary()
model.compile(optimizer="adam",
             loss="categorical_crossentropy",
             metrics=["accuracy"])
model.fit(f_train_images, f_train_cat, epochs=10)
test_loss, test_acc = model.evaluate(f_test_images, f_test_cat)
predictions = model.predict(f_test_images)
def plot_image(i, predictions, f_test_cat, f_test_images):
    prediction, true_label, img = f_test_cat_names[np.argmax(predictions[i])], f_test_cat_names[np.argmax(f_test_cat[i])], f_test_images[i]
    plt.imshow(img)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    if prediction == true_label:
        color = "blue"
    else:
        color = "red"
    
    plt.xlabel("{} \n (pred.: {})".format(true_label, prediction), color=color)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plot_image(i,predictions,f_test_cat,f_test_images)
plt.tight_layout()
plt.show()
