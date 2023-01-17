import tensorflow as tf

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()
training_images = training_images.reshape(60000, 28, 28, 1)

training_images = training_images/ 255.

testing_images = testing_images.reshape(10000, 28, 28, 1)

testing_images = testing_images/ 255.
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(1024, activation=tf.nn.relu),

    tf.keras.layers.Dense(10, activation='softmax')

])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=30, validation_data=(testing_images, testing_labels))