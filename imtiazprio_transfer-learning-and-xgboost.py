# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting tools

from PIL import Image
from PIL.ImageOps import invert

from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, plot_model
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.initializers import he_normal
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import glob
print(os.listdir("../input"))
def loadData(dataset_list ,labels_list=None, size=(64,64), mode='RGB', rng=0):
    """
    Module for loading training or testing data
    
    Parameters:
    dataset_list: List containing paths to datasets
    labels_list: List containing paths to metadata .csv files
    size: Tuple, size to resize images to
    mode: Image channel specifier. Can be any of {'RGB','L','1'}. Please refer to PIL documentaion for more info.
    rng: Number of files to import from each dataset. Specify as 0 to import all.
    
    Returns:
    X: Data Matrix containing images with shape (nsamples,height,width,channels)
    Y: names of images if .csv is not passed, otherwise corresponding digit labels
    
    """
    X = []
    Y = []
    if labels_list is not None: 
        for dataset,labels in zip(dataset_list,labels_list):
            df = pd.read_csv(labels)
            print("loading %s" % dataset)
            for idx,image in enumerate(df.filename):
                if rng and idx == rng:
                    break
                img = Image.open(os.path.join(dataset,image))
                img = img.resize(size)
                img = img.convert(mode)
                if dataset[-1] == 'e':
                    img = invert(img)
                img = np.array(img, dtype=np.uint8())
                X.append(img)
                Y.append(df.digit.iloc[idx])
    else:
        for dataset in dataset_list:
            print("loading %s" % dataset)
            for idx,image in enumerate(os.listdir(dataset)):
                if rng and idx == rng:
                    break
                img = Image.open(os.path.join(dataset,image))
                img = img.resize(size)
                img = img.convert(mode)
                if dataset[-1] == 'e':
                    img = invert(img)
                img = np.array(img, dtype=np.uint8())
                X.append(img)
                Y.append(image.split('/')[-1])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y
def getClassifier(input_shape,dense_neurons=512,dropout_rate=0.2,activation='relu',random_seed=1,lr=0.001,**kwargs):
    """
    Module to build neural network classifier with 1 hidden layer.
    Parameters:
    input_shape: shape of input data without (height, width, channels)
    dense_neurons: Number of hidden layer neurons
    dropout_rate: Amount of activations to be dropped out
    activation: Hidden layer activation function, check keras documentation for details.
    random_seed: Seed for random generator for reproducability
    lr: Learning rate
    
    Returns:
    model: keras model object
    """
    input_layer = Input(shape=input_shape)
    x = Flatten()(input_layer)
    x = Dense(dense_neurons,
              activation=activation,
              kernel_initializer=he_normal(seed=random_seed))(x)
    x = Dropout(rate=dropout_rate,seed=random_seed)(x)
    x = Dense(10,activation='softmax')(x)
    model = Model(input=input_layer,output=x)
    model.compile(optimizer=SGD(lr=lr),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
def generate_pattern(model, layer_name, filter_index, size=150):
    """
    Module to visualize layer weights
    
    Parameters:
    model: keras model object, layers of which to visualize
    layer_name: name of layer to visualize
    filter_index: Index of filter to visualize
    size: size of output image
    
    Returns:
    img: Generated Layer image
    """
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(1000):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        img = input_img_data[0]
        return img
def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)
train_list = sorted(glob.glob(os.path.join('..','input','numta','training-?')))
labels_list = sorted(glob.glob(os.path.join('..','input','numta','training-?.csv')))
test_list = sorted(glob.glob(os.path.join('..','input','numta','testing-*')))

rng = 0
mode = 'RGB'
size = (64,64)

X_train,Y_train = loadData(train_list, labels_list, size=size, mode=mode, rng=rng)
Y_train = to_categorical(Y_train)
X_test,keys_test = loadData(test_list, size=size, mode=mode, rng=0)
# Importing VGG16 model 
model = VGG16(weights=None,include_top=False)
model.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
model.summary()

# Insert Generate Pattern Function Here to Visualize Weights. Try it yourself.

print("Exctracting Features")
featuresTrain = model.predict(X_train,verbose=1)
featuresTest = model.predict(X_test,verbose=1)
del model
del X_train
del X_test
print('Extracted Feature Shape {}'.format(featuresTrain.shape[1:]))
params = dict()
params['dense_neurons'] = 512
params['activation'] = 'relu'
params['random_seed'] = 1
params['dropout_rate'] =0.5
params['lr'] = 0.001

model = getClassifier(featuresTrain.shape[1:],**params)
plot_model(model,'model.png',show_shapes=True)
img = Image.open('model.png','r')
img
if not os.path.exists('modelcheckpnt'):
    os.mkdir('modelcheckpnt')
modelcheckpnt = ModelCheckpoint(filepath=os.path.join('modelcheckpnt','weights.{epoch:04d}-{val_acc:.4f}.hdf5'),
                                monitor='val_acc',save_best_only=True, mode='max')
tensbd = TensorBoard(log_dir='logs',
                     batch_size=64,
                     histogram_freq=50,
                     write_grads=True,
                     write_images=False)
csv_logger = CSVLogger('training.csv')


try:
    model.fit(featuresTrain,Y_train,
              epochs=10,
              batch_size=64,
              verbose=1,
              shuffle=True,
              callbacks = [modelcheckpnt,csv_logger],
              validation_split = 0.2)
except KeyboardInterrupt:
    print('Manually Interrupted')
    
df = pd.read_csv('training.csv')
plt.plot(df['val_acc'],label ='val_acc')
plt.plot(df['acc'], label='train_acc')
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
predictions = model.predict(featuresTest,verbose=1)
predictions = np.argmax(predictions,axis=1)
create_submission(path='submission_transfer_learning_mlp.csv',keys=keys_test,predictions=predictions)
Y_train = np.argmax(Y_train,axis=1)      # Turning Y_train from categorical to singular array
featuresTrain = np.reshape(featuresTrain,newshape=(featuresTrain.shape[0],-1))      # Flatten operation

X_train, X_val, Y_train, Y_val = train_test_split(featuresTrain, Y_train,
                                                   test_size=0.2,
                                                   random_state=params['random_seed'])

model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model.fit(X_train, Y_train, early_stopping_rounds=5,
             eval_set=[(X_val, Y_val)], verbose=True)


featuresTest = np.reshape(featuresTest,newshape=(featuresTest.shape[0],-1))
predictions = model.predict(featuresTest)
create_submission(path='submission_transfer_learning_xgboost.csv',keys=keys_test,predictions=predictions)