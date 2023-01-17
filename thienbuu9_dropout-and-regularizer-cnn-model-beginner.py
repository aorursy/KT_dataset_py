import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizers
DATASET_PATH = "../input/fashionmnist"

train_csv = os.path.join(DATASET_PATH, "fashion-mnist_train.csv")
test_csv = os.path.join(DATASET_PATH, "fashion-mnist_test.csv")
train_df = pd.read_csv(train_csv)
print("Number of entries:", len(train_df))
train_df.head()
test_df = pd.read_csv(test_csv)
print("Number of entries:", len(test_df))
test_df.head()
classes = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}
train_labels = []
train_imgs = []
for idx, row in train_df.iterrows():
    label = row['label']
    img = row.drop('label').values
    img = np.array(img).reshape((28, 28))
    
    train_labels.append(label)
    train_imgs.append(img)
    
train_imgs = tf.convert_to_tensor(train_imgs, tf.float32)
train_labels = tf.convert_to_tensor(train_labels, tf.float32)
test_labels = []
test_imgs = []
for idx, row in test_df.iterrows():
    label = row['label']
    img = row.drop('label').values
    img = np.array(img).reshape((28, 28))
    
    test_labels.append(label)
    test_imgs.append(img)
    
test_imgs = tf.convert_to_tensor(test_imgs, tf.float32)
test_labels = tf.convert_to_tensor(test_labels, tf.float32)
plt.figure(figsize=(10,10))
for i in range(30):
    img = train_imgs[i]
    label = train_labels[i].numpy()
    label_name = classes[label]
    plt.subplot(5,6,i+1)
    plt.imshow(img, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.title(label_name)
plt.figure(figsize=(10,10))
for i in range(30):
    img = test_imgs[i]
    label = test_labels[i].numpy()
    label_name = classes[label]
    plt.subplot(5,6,i+1)
    plt.imshow(img, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    plt.title(label_name)
model = keras.models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(28, 28, 1)),
    # Layer 1
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    # Layer 2
    layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    # Layer 3
    layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.Conv2D(256, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    # Layer 4
    layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.Conv2D(512, 3, padding='same', activation='relu', kernel_regularizer=regularizers.L2(0.001)),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    # Full connected layer
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(10)
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_imgs, train_labels,
                    validation_split=0.3,
                    callbacks=[early_stopping],
                    batch_size=32,
                    epochs=100)
acc = history.history['accuracy']
loss = history.history['loss']

val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
plt.figure(figsize=(15, 6))

plt.subplot(1,2,1)
plt.plot(acc, label='train acc')
plt.plot(val_acc, label='val acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.legend()

plt.show()
test_loss, test_acc = model.evaluate(test_imgs, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = keras.Sequential([
    model,
    layers.Softmax()
])
predictions = probability_model.predict(test_imgs)
plt.figure(figsize=(15,15))
for i in range(30):
    # Get predicted class and probability
    predicted_class = np.argmax(predictions[i])
    predicted_label = classes[predicted_class]
    probability = np.round(np.max(predictions[i]) * 100)
    
    # Get true class
    truth_class = test_labels[i].numpy()
    truth_label = classes[truth_class]
            
    # Prepare display image and its title
    title = "{} {} \n({})".format(predicted_label, probability, truth_label)
    img = test_imgs[i].numpy()

    # Plot image
    plt.subplot(5, 6, i+1)
    plt.imshow(img, cmap='binary')
    if (predicted_class == truth_class):
        plt.xlabel(title, color='blue')
    else:
        plt.xlabel(title, color='red')
    plt.xticks([])
    plt.yticks([])
plt.show()
