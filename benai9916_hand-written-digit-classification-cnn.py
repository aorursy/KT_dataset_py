import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.utils import  plot_model
dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = dataset.load_data()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(set(pd.unique(y_train)))
plt.figure(figsize=(12,8))
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.imshow(x_train[i], cmap='gray')
  plt.tight_layout()
  plt.xlabel('Lable is {}'.format(y_train[i]))
x_train = x_train/255.0
x_train = x_train[..., tf.newaxis]  # adding a channel dimension

x_test = x_test/255.0
x_test = x_test[..., tf.newaxis]
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(500).prefetch(tf.data.experimental.AUTOTUNE)

test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32).shuffle(500).prefetch(tf.data.experimental.AUTOTUNE)
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(6, (5, 5), padding='same', activation='relu', input_shape=(28,28,1)),
  tf.keras.layers.MaxPool2D((2,2)),
  
  tf.keras.layers.Conv2D(16, (5, 5), padding='valid', activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),

  # Flatten

 tf.keras.layers.Flatten(),

 tf.keras.layers.Dense(units=120, activation='relu'),

 tf.keras.layers.Dense(84, activation='relu'),

 tf.keras.layers.Dense(10, activation='softmax')
])
plot_model(model)
# if we use softmax activation in output layer then best fit optimizer is categorical_crossentropy
# for sigmoid activation in output layer then loss will be binary_crossentropy

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),
]
model_his = model.fit(train_data, epochs=30, validation_data=test_data, callbacks=callbacks)
# # evaluate accuracy of the model

test_loss, test_acc = model.evaluate(test_data)
print("accuracy:", test_acc)
# plot the accuracy

plt.plot(model_his.history['accuracy'], label='accuracy')
plt.plot(model_his.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

print('Training accuracy: %f' % model_his.history['accuracy'][-1])
print('Validation accuracy: %f' % model_his.history['val_accuracy'][-1])
