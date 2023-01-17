from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical

# load dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
from tensorflow.keras.applications.vgg19 import VGG19

model = VGG19(
    weights=None, 
    include_top=True, 
    classes=10,
    input_shape=(32,32,3)
)
model.summary()

from tensorflow.keras import optimizers

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor='val_acc', 
    verbose=0, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)
# Train the model
h = model.fit(
    x=X_train,
    y=Y_train,
    validation_split=0.1,
    batch_size=128,
    epochs=50,
    callbacks=[checkpoint],
    verbose=1
)
import matplotlib.pyplot as plt
epoch_nums = range(1, 50+1)
training_loss = h.history["loss"]
validation_loss = h.history["val_loss"]
plt.plot(epoch_nums , training_loss)
plt.plot(epoch_nums , validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training','validation'], loc='upper right')
plt.show()
model.save('VGG19_1.h5')
scores = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
import numpy as np
y_pred_test = model.predict(X_test)
y_pred_test_classes = np.argmax(y_pred_test, axis=1)
y_pred_test_max_probas = np.max(y_pred_test, axis=1)
cols = 8
rows = 2
NUM_CLASSES = 10
# load data
(x_train2, y_train2), (x_test2, y_test2) = cifar10.load_data()
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
fig = plt.figure(figsize=(2 * cols - 1, 3 * rows - 1))
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_test2))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.grid('off')
        ax.axis('off')
        ax.imshow(x_test2[random_index, :])
        pred_label =  cifar10_classes[y_pred_test_classes[random_index]]
        pred_proba = y_pred_test_max_probas[random_index]
        true_label = cifar10_classes[y_test2[random_index, 0]]
        ax.set_title("pred: {}\nscore: {:.3}\ntrue: {}".format(
               pred_label, pred_proba, true_label
        ))
plt.show()
