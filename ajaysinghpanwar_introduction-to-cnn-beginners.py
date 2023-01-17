# Importing the libraries

import tensorflow as tf



from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
# Let's download the data

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values between 0 and 1

train_images, test_images = train_images/255.0, test_images/255.0
train_images.shape
# Let's plot some images to check the data

class_names = ['airplane', 'automobile', 'bird', 'cat', 'lion',

               'dog', 'frog', 'horse', 'ship', 'truck']



plt.figure(figsize = (12,12))



for i in range(25):

    plt.subplot(5, 5, i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(train_images[i], cmap = plt.cm.binary)

    plt.xlabel(class_names[train_labels[i][0]])

plt.show()
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))

model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))

model.add(layers.MaxPool2D(2,2))

model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
# Let's check our model's architecture till now

model.summary()
# Let's now add flatten and dense layers

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10))

# Complete architecture of the model

model.summary()
model.compile(optimizer = 'adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),

             metrics = ['accuracy'])



history = model.fit(train_images, train_labels, epochs = 10, validation_data = (test_images, test_labels))
plt.plot(history.history['accuracy'], label = 'accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.ylim([0.5, 1])

plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)