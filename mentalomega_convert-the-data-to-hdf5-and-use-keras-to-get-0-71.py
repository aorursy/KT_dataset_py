# I use this code in my computer in order to convert, it should be work in your computer 
from glob import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
pf = pd.read_csv("trainLabels.csv")
pf

c2i = {category:id for id,category in enumerate(set(pf["label"]))}
i2c = np.asarray(list(set(pf["label"])))
np.save("int2categroy.npy",i2c)
c2i,i2c

# build the one hot matrix
train_label = np.zeros([len(pf["label"]),len(i2c)],dtype="uint8")
for id, label in enumerate(pf["label"]):
    train_label[id][c2i[label]] = 1
train_label

def read_image(path):
    image = keras.preprocessing.image.load_img(path)
    return keras.preprocessing.image.img_to_array(image)

train_data = np.zeros([len(glob("train/*")),32,32,3],dtype="uint8")
for p in tqdm(glob("train/*")):
    id = int(p.split("/")[-1].split(".")[0]) - 1 
    train_data[id] = read_image(p)
    
# convert to train.hdf5
import h5py
train_f = h5py.File("train.hdf5","w")
train_f.create_dataset("train_data",data=train_data)
train_f.create_dataset("train_label",data=train_label)

train_f.keys(),train_f["train_data"],train_f["train_label"]

plt.imshow(train_f["train_data"][0]) # check the label

train_f.close()

# convert to train.hdf5
import h5py
test_f = h5py.File("test.hdf5","w")
test_f.create_dataset("test_data",data=test_data)

test_f.keys(),test_f["test_data"]

test_f.close()
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import h5py
import matplotlib.pyplot as plt
train_f = h5py.File("../input/cifar10-compete/train.hdf5","r")
train_data = train_f["train_data"]
train_label = train_f["train_label"]
train_data,train_label
plt.imshow(train_data[0])
import tensorflow as tf
import keras 
from keras.layers import *
from keras.models import Model
from keras.applications.densenet import preprocess_input
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
# https://github.com/keras-team/keras/tree/master/examples
# data augment
# if we dont use the augment, the validation accuracy will be bad
datagen = ImageDataGenerator(
    # set input mean to 0 over the dataset
    featurewise_center=False,
    # set each sample mean to 0
    samplewise_center=False,
    # divide inputs by std of dataset
    featurewise_std_normalization=False,
    # divide each input by its std
    samplewise_std_normalization=False,
    # apply ZCA whitening
    zca_whitening=False,
    # epsilon for ZCA whitening
    zca_epsilon=1e-06,
    # randomly rotate images in the range (deg 0 to 180)
    rotation_range=0,
    # randomly shift images horizontally
    width_shift_range=0.1,
    # randomly shift images vertically
    height_shift_range=0.1,
    # set range for random shear
    shear_range=0.,
    # set range for random zoom
    zoom_range=0.,
    # set range for random channel shifts
    channel_shift_range=0.,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    # value used for fill_mode = "constant"
    cval=0.,
    # randomly flip images
    horizontal_flip=True,
    # randomly flip images
    vertical_flip=False,
    # set rescaling factor (applied before any other transformation)
    rescale=1./255,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)
datagen.fit(train_data)
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
print('Running on TPU: ', tpu.cluster_spec().as_dict()['worker'])

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
print('Replicas: ', strategy.num_replicas_in_sync)
with strategy.scope():
    input = Input(train_data[0].shape)
    img_net = DenseNet121(weights=None, input_shape = train_data[0].shape, include_top=False)
#     if using transfer learning then fixing the img_net 
#     img_net.trainable = False
# we can adding a normalize layer after input layer, but I have normalize in ImageDataGenerator
# with strategy.scope():
#     def normalize(x):
#         return x/255
# #         return preprocess_input(x)
#     Normalize = Lambda(normalize)
with strategy.scope():
    x = input
#     x = Normalize(x)
    x = img_net(x)
    x = Flatten()(x)
    x = Dense(512,activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(10,activation="softmax")(x)
    output = x
    model = Model(input,output)
    model.summary()
with strategy.scope():
    model.compile("adam",loss="categorical_crossentropy",metrics=['accuracy'])
with strategy.scope():
    history = model.fit_generator(datagen.flow(train_data,train_label,batch_size=256),epochs=25) 
# if you want to get a better result 
# https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
# https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
history_df = pd.DataFrame(history.history)
history_df[["loss"]].plot()
history_df[["accuracy"]].plot()
model.save_weights("cifar10_Densenet121_.weight")
# Prediction
test_f = h5py.File("../input/cifar10-compete/test.hdf5","r")
test_data = test_f["test_data"]
test_data
test_label = np.argmax(model.predict(test_data[:]/255),-1) # test_data[:]/255 will comsuming lots of memory. In this time is ok
# so adding a normalize layer after input layer can prevent it
# convert the int to categroy
i2c = np.load("../input/cifar10-compete/int2categroy.npy")
i2c
test_label = [i2c[i] for i in test_label]
answers = pd.DataFrame({"id":np.arange(1,300000+1),"label":test_label})
answers
answers.to_csv("answers.csv",index=0) # Finally, download the answers.csv and summit it. 