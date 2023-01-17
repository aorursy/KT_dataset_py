# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pathlib
Cover = pathlib.Path("../input/alaska2-image-steganalysis/Cover")
JUNIWARD = pathlib.Path("../input/alaska2-image-steganalysis/JUNIWARD")
JMiPOD = pathlib.Path("../input/alaska2-image-steganalysis/JMiPOD")
UERD = pathlib.Path("../input/alaska2-image-steganalysis/UERD")

# save it as binary data with (image , lable), where if paracsitized ==1 , else 0
from PIL import Image
import h5py
from tqdm import tqdm
import numpy as np

folders = [Cover , JUNIWARD]
filepath= []
for folder in folders:
    f = list(folder.glob('*.jpg'))[:250]
    filepath.extend(f)

CLASS_NAMES = np.array(["Cover","JUNIWARD"])
print(CLASS_NAMES)

with h5py.File('AlaskaBinary1.h5', 'w') as hf:
  print("creating trainX")
  trainX = []
  trainY = []
  np.random.shuffle(filepath)
  for i in tqdm(filepath, ascii = True, desc = "Train data"):
    img = Image.open(i)
    #img  = img.resize((,100))
    img = np.asarray(img)
    img = img.astype('float32')
    # normalize to the range 0-1
    img /= 255.0
    name = i.parts
    lable = True
    if name[3] == 'Cover':
       lable = False
    trainX.append(img)
    trainY.append(lable)
  trainX = np.asarray(trainX)
  trainY = np.asarray(trainY)
  print(trainX.shape)
  print(trainX[0].shape)
  print("Train done")
  hf.create_dataset("trainX",  data=trainX)
  hf.create_dataset("trainY",  data=trainY)
  del trainX, trainY
print("All done")
import numpy as np
import h5py
# import the necessary packages
import tensorflow as tf
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# plot the training loss and accuracy
def plot_graph(H):
  list_loss = H.history["val_loss"]
  min_val = min(list_loss)
  min_index = list_loss.index(min_val)
  t_acc = H.history["accuracy"][min_index]
  v_acc = H.history["val_accuracy"][min_index]
  print(f"The model {H.model.name} has the lowest val_loss {min_val} at epoch {min_index+1} with \ntrain accuracy of {t_acc} and validation accuracy {v_acc}." )
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
  plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend()
  plt.savefig(f"{H.model.name}_{v_acc}.png")

def loadDataH5():
    with h5py.File('AlaskaBinary1.h5','r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
#         valX = np.array(hf.get('valX'))
#         valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        #print (valX.shape,valY.shape)
    return trainX, trainY

X, Y = loadDataH5()
from sklearn.model_selection import train_test_split
trainX,testX, trainY, testY = train_test_split(X, Y, test_size=0.20, random_state=13)
import os

def runmodel(model):
  # defining common params.
  NUM_EPOCHS = 100
  opt = tf.keras.optimizers.SGD(lr=0.01)
  # initialize the optimizer and model
  print("Compiling model...")
  print (model.summary())
  tmpdir = f"./{model.name}"
  os.makedirs(tmpdir, exist_ok=True)
  fname = "checkpoint.hdf5"
  f = os.path.join(tmpdir, fname)
  checkpoint = tf.keras.callbacks.ModelCheckpoint(f, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
  model.compile(loss="BinaryCrossentropy", optimizer=opt,
    metrics=["accuracy"])
  # train the network
  print("Training network...")
  H = model.fit(trainX, trainY,batch_size = 32, validation_data=(testX, testY),steps_per_epoch=len(trainX) / 32,callbacks=[checkpoint], epochs=NUM_EPOCHS)
  plot_graph(H)
trainY[:100]
def LeNet_5(width, height, depth):
  # alternate maxpooling
  # initialize the model along with the input shape to be "channels last"
  model = tf.keras.Sequential(name='LeNet_5-Esem3') 
  inputShape = (height, width, depth)
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=inputShape))
  model.add(tf.keras.layers.MaxPool2D(strides=2))
  model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
  model.add(tf.keras.layers.MaxPool2D(strides=2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dense(84, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  return model
runmodel(LeNet_5(width=512, height=512, depth=3))
import pathlib
Cover = pathlib.Path("../input/alaska2-image-steganalysis/Cover")
JUNIWARD = pathlib.Path("../input/alaska2-image-steganalysis/JUNIWARD")
JMiPOD = pathlib.Path("../input/alaska2-image-steganalysis/JMiPOD")
UERD = pathlib.Path("../input/alaska2-image-steganalysis/UERD")

# save it as binary data with (image , lable), where if paracsitized ==1 , else 0
from PIL import Image
import h5py
from tqdm import tqdm
import numpy as np

folders = [Cover , JMiPOD]
filepath= []
for folder in folders:
    f = list(folder.glob('*.jpg'))[:250]
    filepath.extend(f)

CLASS_NAMES = np.array(["Cover","JMiPOD"])
print(CLASS_NAMES)

with h5py.File('AlaskaBinary2.h5', 'w') as hf:
  print("creating trainX")
  trainX = []
  trainY = []
  np.random.shuffle(filepath)
  for i in tqdm(filepath, ascii = True, desc = "Train data"):
    img = Image.open(i)
    #img  = img.resize((,100))
    img = np.asarray(img)
    img = img.astype('float32')
    # normalize to the range 0-1
    img /= 255.0
    name = i.parts
    lable = True
    if name[3] == 'Cover':
       lable = False
    trainX.append(img)
    trainY.append(lable)
  trainX = np.asarray(trainX)
  trainY = np.asarray(trainY)
  print(trainX.shape)
  print(trainX[0].shape)
  print("Train done")
  hf.create_dataset("trainX",  data=trainX)
  hf.create_dataset("trainY",  data=trainY)
  del trainX, trainY
print("All done")

def loadDataH5(name):
    with h5py.File(name,'r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
#         valX = np.array(hf.get('valX'))
#         valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        #print (valX.shape,valY.shape)
    return trainX, trainY

X, Y = loadDataH5('AlaskaBinary2.h5')

trainX,testX, trainY, testY = train_test_split(X, Y, test_size=0.20, random_state=13)

def LeNet_5(width, height, depth):
  # alternate maxpooling
  # initialize the model along with the input shape to be "channels last"
  model = tf.keras.Sequential(name='LeNet_5-Esem2') 
  inputShape = (height, width, depth)
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=inputShape))
  model.add(tf.keras.layers.MaxPool2D(strides=2))
  model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
  model.add(tf.keras.layers.MaxPool2D(strides=2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dense(84, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  return model
runmodel(LeNet_5(width=512, height=512, depth=3))


os.remove("./AlaskaBinary2.h5")
import pathlib
Cover = pathlib.Path("../input/alaska2-image-steganalysis/Cover")
JUNIWARD = pathlib.Path("../input/alaska2-image-steganalysis/JUNIWARD")
JMiPOD = pathlib.Path("../input/alaska2-image-steganalysis/JMiPOD")
UERD = pathlib.Path("../input/alaska2-image-steganalysis/UERD")

# save it as binary data with (image , lable), where if paracsitized ==1 , else 0
from PIL import Image
import h5py
from tqdm import tqdm
import numpy as np

folders = [Cover , UERD]
filepath= []
for folder in folders:
    f = list(folder.glob('*.jpg'))[:250]
    filepath.extend(f)

CLASS_NAMES = np.array(["Cover","UERD"])
print(CLASS_NAMES)

with h5py.File('AlaskaBinary3.h5', 'w') as hf:
  print("creating trainX")
  trainX = []
  trainY = []
  np.random.shuffle(filepath)
  for i in tqdm(filepath, ascii = True, desc = "Train data"):
    img = Image.open(i)
    #img  = img.resize((,100))
    img = np.asarray(img)
    img = img.astype('float32')
    # normalize to the range 0-1
    img /= 255.0
    name = i.parts
    lable = True
    if name[3] == 'Cover':
       lable = False
    trainX.append(img)
    trainY.append(lable)
  trainX = np.asarray(trainX)
  trainY = np.asarray(trainY)
  print(trainX.shape)
  print(trainX[0].shape)
  print("Train done")
  hf.create_dataset("trainX",  data=trainX)
  hf.create_dataset("trainY",  data=trainY)
  del trainX, trainY
print("All done")

def loadDataH5(name):
    with h5py.File(name,'r') as hf:
        trainX = np.array(hf.get('trainX'))
        trainY = np.array(hf.get('trainY'))
#         valX = np.array(hf.get('valX'))
#         valY = np.array(hf.get('valY'))
        print (trainX.shape,trainY.shape)
        #print (valX.shape,valY.shape)
    return trainX, trainY

X, Y = loadDataH5('AlaskaBinary3.h5')

trainX,testX, trainY, testY = train_test_split(X, Y, test_size=0.20, random_state=13)

def LeNet_5(width, height, depth):
  # alternate maxpooling
  # initialize the model along with the input shape to be "channels last"
  model = tf.keras.Sequential(name='LeNet_5-Esem1') 
  inputShape = (height, width, depth)
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=inputShape))
  model.add(tf.keras.layers.MaxPool2D(strides=2))
  model.add(tf.keras.layers.Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
  model.add(tf.keras.layers.MaxPool2D(strides=2))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(256, activation='relu'))
  model.add(tf.keras.layers.Dense(84, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  return model
runmodel(LeNet_5(width=512, height=512, depth=3))


os.remove("./AlaskaBinary3.h5")
import pandas as pd
from kaggle_datasets import KaggleDatasets

# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# load submission
sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')


# Data access
GCS_DS_PATH = '/kaggle/input/alaska2-image-steganalysis'

def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))

test_paths = append_path('Test')(sub.Id.values)
print(test_paths)

def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    return image


test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(32)
)
# Creating ensemble
model1 = tf.keras.models.load_model('./LeNet_5-Esem1/checkpoint.hdf5')
model2 = tf.keras.models.load_model('./LeNet_5-Esem2/checkpoint.hdf5')
model3 = tf.keras.models.load_model('./LeNet_5-Esem3/checkpoint.hdf5')
y_pred1 = model1.predict(test_dataset, verbose=1)
y_pred2= model2.predict(test_dataset, verbose=1)
y_pred3= model3.predict(test_dataset, verbose=1)

sub.Label = (y_pred1+y_pred2+y_pred3)/3
sub.to_csv('submission.csv', index=False)
sub.head()
                    