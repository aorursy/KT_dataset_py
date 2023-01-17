import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense,Dropout,Flatten,BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

import cv2
train_data = "../input/dog-breed-identification/train"
label_csv = "../input/dog-breed-identification/labels.csv"
label_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")
label_csv["path"] = '../input/dog-breed-identification/train/'+label_csv["id"]+".jpg"
label_csv.head()
#Checking number of dog breeds

label_csv["breed"].nunique()
x = np.zeros((label_csv.shape[0], 128, 128, 3))

for i, index in tqdm(enumerate(label_csv["path"])):
#     image_path = os.path.join(data_dir+"/images", all_keys[index])
    image = tf.keras.preprocessing.image.load_img(index, target_size=(128,128))

    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    x[i] = arr
#Converting the breed column into categorical class

label_csv.breed = pd.Categorical(pd.factorize(label_csv.breed)[0])
y = to_categorical(label_csv["breed"],num_classes=120)
from pylab import rcParams
rcParams['figure.figsize'] = 25, 5

label_csv["breed"].value_counts().plot.bar(color=["r","orange","y","b","green","pink"])

plt.title("Total count of different dog breed")
plt.ylabel('Number')
plt.xlabel('Class')
plt.show()
plt.figure(figsize=(15,15))

start_index = 50

for i in range(16):
    plt.subplot(4,4, i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    image = label_csv["path"][start_index+i]

    plt.imshow(cv2.imread(image))
    plt.tight_layout()

plt.show()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=2)
# X_train, X_test=X_train/255,X_test/255
tf.keras.backend.clear_session()

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU
with tpu_strategy.scope():
    mnet = MobileNetV2(include_top = False, pooling="max", weights = "imagenet" ,input_shape=(128,128,3))
    model = Sequential([mnet,
                        BatchNormalization(),
                        Dropout(0.6),
                        Dense(120, activation="softmax")])

    model.layers[0].trainable = False

    model.compile(loss="categorical_crossentropy", metrics = "accuracy", optimizer="adam")
model.summary()
#We will reduce the learning rate when then accuracy not increase for 2 steps

reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.2,patience=2, min_lr=1e-7, verbose=1, mode="max" )

#To prevent over fitting we will stop the learning after 3 epochs and val_loss value not decreased

stop = EarlyStopping(monitor='val_accuracy', patience=3, mode="max" )
history = model.fit(X_train, y_train, epochs=20,callbacks=[stop,reduce_lr], batch_size=32, validation_data = (X_test, y_test))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, 20, 1))
ax1.set_yticks(np.arange(0, 1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, 20, 1))
ax1.set_yticks(np.arange(0, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
#Creating an array of predicted test images

predictions = model.predict(X_test)
plt.figure(figsize=(15,15))

start_index = 50

for i in range(16):
  plt.subplot(4,4, i+1)
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  preds = np.argmax(predictions[[start_index+i]])
    
  gt = np.argmax(y_test[start_index+i])

  col = "g"
  if preds != gt:
    col ="r"

  plt.xlabel('i={}, pred={}, gt={}'.format(start_index+i,preds,gt),color=col)
  plt.imshow(X_test[start_index+i])
  plt.tight_layout()

plt.show()