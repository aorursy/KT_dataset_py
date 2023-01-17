import tensorflow as tf

mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#x_train, x_test = x_train / 255.0, x_test / 255.0
print("Input Shape"+ str(x_train.shape))
import matplotlib.pyplot as plt
import numpy as np
#transformati imaginea
print("shape imagine = "+ str(x_imagine.shape))
plt.imshow(x_imagine)


predictions = model(x_train[:1]).numpy()
predictions
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.fit(x_train, y_train, epochs=5)

x_imagine = x_test[:1].reshape((28,28))
print("shape imagine = "+ str(x_imagine.shape))

y_pred = np.argmax(tf.nn.softmax(model(x_test[:1]).numpy()).numpy())
print(y_pred)
model.evaluate(x_train,  y_train, verbose=2)
model.evaluate(x_test,  y_test, verbose=2)