import sklearn.datasets
import sklearn.model_selection
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
def create_dataset(n_samples, noise=0):
    x, y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.4)
    return (x_train, x_test, y_train, y_test)
x_train, x_test, y_train, y_test = create_dataset(n_samples=200)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train);
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'), # input_shape is not necessary, but can be useful for debugging.
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50).batch(10)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(50).batch(10)

model.fit(train_ds, validation_data=test_ds, epochs=500)
def plot_decision_boundary(model, x_train, y_train, x_test, y_test):
    """ 
        Make a plot of locations where the model predicts 0 vs 1.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5));
    xx, yy = np.meshgrid(np.arange(x_train[:,0].min()*1.1, x_train[:,0].max()*1.1, 0.02),
                         np.arange(x_train[:,1].min()*1.1, x_train[:,1].max()*1.1, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = np.around(np.reshape(Z, xx.shape))
    axes[0].contourf(xx, yy, z, alpha=.8);
    axes[1].contourf(xx, yy, z, alpha=.8);
    axes[0].scatter(x_train[:, 0], x_train[:, 1], c=y_train);
    axes[1].scatter(x_test[:, 0], x_test[:, 1], c=y_test);
    axes[0].set_title("Training data")
    axes[1].set_title("Test data")
    return (fig, axes)
plot_decision_boundary(model, x_train, y_train, x_test, y_test);
x_train, x_test, y_train, y_test = create_dataset(n_samples=100, noise=0.6)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train);
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(2,), activation='relu'), # input_shape is not necessary, but can be useful for debugging.
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50).batch(10)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(50).batch(10)

model.fit(train_ds, validation_data=test_ds, epochs=500)

plot_decision_boundary(model, x_train, y_train, x_test, y_test);
n_layers = 7
layer_width = 128
l1 = 0.0
l2 = 0.0
dropout_frac = 0.0

model = tf.keras.Sequential()
for i in range(n_layers):
    model.add(tf.keras.layers.Dense(layer_width, activation='relu',
                    kernel_regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(tf.keras.layers.Dropout(dropout_frac))

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50).batch(10)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(50).batch(10)

model.fit(train_ds, validation_data=test_ds, epochs=500)
plot_decision_boundary(model, x_train, y_train, x_test, y_test);
