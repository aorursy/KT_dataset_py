import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Activation, Flatten, Dense, AveragePooling2D
from sklearn.utils import shuffle
tf.test.is_gpu_available()
tf.test.gpu_device_name()
def convert_array(df):
    list_images = []
    for i in range (df.shape[0]):
        images1 = df.loc[i,:]
        lst = images1.values.tolist()
        lst = np.asarray(lst)
        new_arr = lst.reshape(28,28,1)
        new_lst = new_arr.tolist()
        list_images.append(new_lst)
    return np.asarray(list_images)
def network(width, height):
    model = keras.Sequential()
    inputS = (height, width, 1)
    chanDim = -1
    model.add(Conv2D(8, (5, 5), padding="same",input_shape=inputS))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(16, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.summary()
    model.compile(loss="sparse_categorical_crossentropy", optimizer='Adam',  metrics=["accuracy"])
    return model
    
def evaluation_vis(History, string, epochs):
    N = np.arange(0, epochs)
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 8))
    plt.plot(N, History.history[string], label="train " + string)
    plt.plot(N, History.history["val_" + string], label="val " + string)
    plt.title("Training and Validation " + string)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Val" + string)
    plt.legend(loc="lower left")

train_df = pd.read_csv('../input/digit-recognizer/train.csv')
test_df = pd.read_csv('../input/digit-recognizer/test.csv')

#plot labels
train_df.groupby('label').size().plot(kind='bar',figsize = (14,8),color=[ 'blue','red', 'green','skyblue','olive','mediumorchid', 'orange', 'cyan', 'pink','darkgoldenrod'])
plt.title("Number of Training samples")
plt.xlabel("Labels")
plt.ylabel("Samples")
plt.show()
y = train_df.label
y = np.asarray(y)
images = train_df.loc[:, train_df.columns != 'label']
x = convert_array(images)
test = convert_array(test_df)
print("Training set size: ", x.shape)
print("Test set size: ", test.shape)
x, y = shuffle(x, y)
split_size = int(x.shape[0] * 0.9)
x_train, x_val = x[:split_size], x[split_size:]
y_train, y_val = y[:split_size], y[split_size:]

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose = 1,
                              patience=2, min_lr=0.00001)
eps = 60
bs = 128

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.2,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=False,rescale=0.6, brightness_range= [0.15,0.7], fill_mode="nearest")

aug.fit(x_train)

model = network(28, 28)
History = model.fit_generator(aug.flow(x_train, y_train, batch_size = bs),
 validation_data = (x_val, y_val), steps_per_epoch = len(x_train) // bs,
 epochs = eps, callbacks=[reduce_lr])
score = model.evaluate(x_val, y_val, verbose=0)
print("Test Loss: {}".format(score[0]))
print("Test Accuracy: {}".format(score[1]))
evaluation_vis(History, 'loss', eps)
evaluation_vis(History, 'accuracy', eps)
model.predict(test)
y_hat = model.predict_classes(test, verbose=0)
pr_class = model.predict_classes(x_val)
fig,axes = plt.subplots(5,5,figsize=(13,13))
axes = axes.ravel()
for i in np.arange(0,25):
    axes[i].imshow(x_val[i].reshape(28,28))
    value_pr = pr_class[i]
    value_true = y_val[i]
    axes[i].set_title("Prediction = {}\n True={}".format(value_pr, value_true ),fontsize=8)
    axes[i].axis("off")
pr_class = model.predict_classes(test)
fig,axes = plt.subplots(5,5,figsize=(13,13))
axes = axes.ravel()
for i in np.arange(0,25):
    axes[i].imshow(test[i].reshape(28,28))
    value_pr = pr_class[i]
    axes[i].set_title("Prediction = {}".format(value_pr),fontsize=8)
    axes[i].axis("off")
submission = pd.Series(y_hat, name="Label")
submission = pd.concat([pd.Series(range(1,28001), name ="ImageId"), submission], axis = 1)
submission.to_csv("submission.csv", index=False)
