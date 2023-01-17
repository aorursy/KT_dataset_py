import tensorflow as tf

mnist = tf.keras.datasets.mnist #28*28 image of hand written digit 0-9
(train_x, y_train), (x_test,y_test) = mnist.load_data()

train_x = tf.keras.utils.normalize(train_x)

x_test = tf.keras.utils.normalize(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, y_train, epochs=3)
import matplotlib.pyplot as plt
plt.imshow(train_x[0])
#plt.show()

#print(train_x[0])
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
model.save('epic_num-reader.model')
new_model = tf.keras.models.load_model('epic_num-reader.model')
