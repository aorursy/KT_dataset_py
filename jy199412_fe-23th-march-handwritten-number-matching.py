import tensorflow as tf
(x_train, y_train),(x_test, y_test)=tf.keras.datasets.mnist.load_data() #x is a feature/input, y is a label
x_train.shape
import matplotlib.pyplot as plt

plt.imshow(x_train[0])

print(y_train[0])
plt.imshow(x_train[110])

print(y_train[110])
x_train=x_train/255

x_test=x_test/255



model=tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(512,activation=tf.nn.relu),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(10,activation='softmax')

])



model.compile(loss='sparse_categorical_crossentropy',

             optimizer='adam',

             metrics=['accuracy'])



model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test)
model.summary()