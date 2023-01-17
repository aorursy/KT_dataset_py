import tensorflow as tf
import numpy as np
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)
def Build_network():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Conv2D(24,3))
  model.add(tf.keras.layers.MaxPool2D())
  model.add(tf.keras.layers.Conv2D(36,3))
  model.add(tf.keras.layers.MaxPool2D())
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(200, activation="relu"))
  model.add(tf.keras.layers.Dense(100, activation="relu"))
  model.add(tf.keras.layers.Dense(10, activation="softmax"))
  return model
model = Build_network()
model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, batch_size=64, epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("accuracy: ", test_acc)
model.save("MNIST_99acc.h5")
import matplotlib.pyplot as plt
predictions = model.predict(test_images)
n = 0
print("Prediction: ", np.argmax(predictions[n]), "Label: ", test_labels[n])
plt.imshow(test_images[n,:,:,0])
plt.show()
n = 52
print("Prediction", np.argmax(predictions[n]), "Label: ", test_labels[n])
plt.imshow(test_images[n,:,:,0])
plt.show()
n = 528
print("Prediction", np.argmax(predictions[n]), "Label: ", test_labels[n])
plt.imshow(test_images[n,:,:,0])
plt.show()
