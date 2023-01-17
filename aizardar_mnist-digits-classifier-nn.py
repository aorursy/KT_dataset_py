import tensorflow as tf

print(tf.__version__)
# Let's load the dataset



mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
import matplotlib.pyplot as plt

plt.imshow(training_images[0])

print(training_labels[0])

print(training_images[0])
training_images  = training_images / 255.0

test_images = test_images / 255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 

                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 

                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer = tf.keras.optimizers.Adam(),

              loss = 'sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)



print(classifications[0])

print(test_labels[0])
import tensorflow as tf

print(tf.__version__)



mnist = tf.keras.datasets.mnist



(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()



training_images = training_images/255.0

test_images = test_images/255.0



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),

                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy')



model.fit(training_images, training_labels, epochs=5)



model.evaluate(test_images, test_labels)



classifications = model.predict(test_images)



print(classifications[0])

print(test_labels[0])
import tensorflow as tf

print(tf.__version__)



mnist = tf.keras.datasets.mnist



(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()



training_images = training_images/255.0

test_images = test_images/255.0



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),

                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu),

                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])



model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy')



model.fit(training_images, training_labels, epochs=15)



model.evaluate(test_images, test_labels)



classifications = model.predict(test_images)



print(classifications[0])

print(test_labels[0])