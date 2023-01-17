# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from tqdm import tqdm

# for reading and displaying images
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
%matplotlib inline

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# torchvision for pre-trained models
from torchvision import models
train = pd.read_csv('/kaggle/input/train_SOaYf6m/train.csv')
train.head()
train_img = []
for img_name in tqdm(train['image_names']):
    # defining the image path
    image_path = '/kaggle/input/train_SOaYf6m/images/' + img_name
    # reading the image
    img = imread(image_path)
    # normalizing the pixel values
    img = img/255
    # resizing the image to (224,224,3)
    img = resize(img, output_shape=(224,224,3), mode='constant', anti_aliasing=True)
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    train_img.append(img)

# converting the list to numpy array
train_x = np.array(train_img)
train_x.shape
train_y = train['emergency_or_not'].values
train_y.shape
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, random_state = 13, stratify=train_y)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
train_x = train_x.reshape(1481, 3, 224, 224)
train_x  = torch.from_numpy(train_x)

# converting the target into torch format
train_y = train_y.astype(int)
train_y = torch.from_numpy(train_y)

val_x = val_x.reshape(165, 3, 224, 224)
val_x  = torch.from_numpy(val_x)

# converting the target into torch format
val_y = val_y.astype(int)
val_y = torch.from_numpy(val_y)

# shape of validation data
val_x.shape, val_y.shape
# loading the pretrained model
model = models.vgg16_bn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Add on classifier
model.classifier[6] = Sequential(
                      Linear(4096, 2))
for param in model.classifier[6].parameters():
    param.requires_grad = True
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
# batch_size
batch_size = 16

# extracting features for train data
data_x = []
label_x = []

inputs,labels = train_x, train_y

for i in tqdm(range(int(train_x.shape[0]/batch_size)+1)):
    input_data = inputs[i*batch_size:(i+1)*batch_size]
    label_data = labels[i*batch_size:(i+1)*batch_size]
    input_data , label_data = Variable(input_data).cuda(),Variable(label_data).cuda()
    x = model.features(input_data)
    data_x.extend(x.data.cpu().numpy())
    label_x.extend(label_data.data.cpu().numpy())
data_y = []
label_y = []

inputs,labels = val_x, val_y

for i in tqdm(range(int(val_x.shape[0]/batch_size)+1)):
    input_data = inputs[i*batch_size:(i+1)*batch_size]
    label_data = labels[i*batch_size:(i+1)*batch_size]
    input_data , label_data = Variable(input_data.cuda()),Variable(label_data.cuda())
    x = model.features(input_data)
    data_y.extend(x.data.cpu().numpy())
    label_y.extend(label_data.data.cpu().numpy())
x_train  = torch.from_numpy(np.array(data_x))
x_train = x_train.view(x_train.size(0), -1)
y_train  = torch.from_numpy(np.array(label_x))
x_val  = torch.from_numpy(np.array(data_y))
x_val = x_val.view(x_val.size(0), -1)
y_val  = torch.from_numpy(np.array(label_y))
import torch.optim as optim

# specify loss function (categorical cross-entropy)
criterion = CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.0005)
batch_size = 16

# number of epochs to train the model
n_epochs = 30

for epoch in tqdm(range(1, n_epochs+1)):

    # keep track of training and validation loss
    train_loss = 0.0
        
    permutation = torch.randperm(x_train.size()[0])

    training_loss = []
    for i in range(0,x_train.size()[0], batch_size):

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = x_train[indices], y_train[indices]
        
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        
        optimizer.zero_grad()
        # in case you wanted a semi-full example
        outputs = model.classifier(batch_x)
        loss = criterion(outputs,batch_y)

        training_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        
    training_loss = np.average(training_loss)
    print('epoch: \t', epoch, '\t training loss: \t', training_loss)
# prediction for validation set
prediction_val = []
target_val = []
permutation = torch.randperm(x_val.size()[0])
for i in tqdm(range(0,x_val.size()[0], batch_size)):
    indices = permutation[i:i+batch_size]
    batch_x, batch_y = x_val[indices], y_val[indices]

    if torch.cuda.is_available():
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

    with torch.no_grad():
        output = model.classifier(batch_x.cuda())

    softmax = torch.exp(output).cpu()
    prob = list(softmax.numpy())
    predictions = np.argmax(prob, axis=1)
    prediction_val.append(predictions)
    target_val.append(batch_y)
    
# validation accuracy
accuracy_val = []
for i in range(len(prediction_val)):
    accuracy_val.append(accuracy_score(target_val[i].cpu(),prediction_val[i]))
    
print('validation accuracy: \t', np.average(accuracy_val))



from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.vgg16 import VGG16

HEIGHT = 224
WIDTH = 224

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
#base_model = VGG16(weights='imagenet', include_top=False,input_shape=(HEIGHT, WIDTH, 3))
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras.applications import VGG16
from keras.models import Model
from keras import layers
from tensorflow.keras import optimizers
image_size = 224
input_shape = (image_size, image_size, 3)
pre_trained_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True
    
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(2, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=5e-4),
              metrics=['accuracy'])

model.summary()
train = pd.read_csv('/kaggle/input/train_SOaYf6m/train.csv')
train['emergency_or_not'] = train['emergency_or_not'].astype(str)
train.head()
from keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "/kaggle/input/train_SOaYf6m/images/"
#TRAIN_DIR = "/kaggle/input/completedata/All/"
#VAL_DIR = "/kaggle/input/validation/Valid/"
HEIGHT = 224
WIDTH = 224
BATCH_SIZE = 16
transformation_ratio = .05
train_datagen =  ImageDataGenerator(
    rescale=1. / 255,
      preprocessing_function=preprocess_input,
      rotation_range=transformation_ratio,
       shear_range=transformation_ratio,
     zoom_range=transformation_ratio,
     cval=transformation_ratio,
    horizontal_flip=True,
      vertical_flip=True,
        validation_split = 0.2)



train_generator=train_datagen.flow_from_dataframe(dataframe=train,directory="/kaggle/input/train_SOaYf6m/images/",x_col="image_names",y_col="emergency_or_not",subset="training",
                                            target_size=[HEIGHT, WIDTH],   batch_size=BATCH_SIZE,class_mode="categorical")
valid_generator=train_datagen.flow_from_dataframe(dataframe=train,directory="/kaggle/input/train_SOaYf6m/images/",x_col="image_names",y_col="emergency_or_not",
                                                       subset="validation",target_size=[HEIGHT, WIDTH],   batch_size=BATCH_SIZE,class_mode="categorical")

NUM_EPOCHS = 30
BATCH_SIZE = 16
num_train_images = 1317
nb_validation_samples = 329
history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE,
                                       validation_data=valid_generator,
                        validation_steps=nb_validation_samples // BATCH_SIZE,
                                       shuffle=True)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,26)
plt.plot(epochs, loss_train[0:25], 'g', label='Training loss')
plt.plot(epochs, loss_val[0:25], 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = range(1,26)
plt.plot(epochs, loss_train[0:25], 'g', label='Training accuracy')
plt.plot(epochs, loss_val[0:25], 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
path = '/kaggle/input/'
test_path = path+'Test Images/'
def genelable(model, string):
    id_list=list(test.image_names)
    label=[]
    for iname in id_list:
        if(int(iname[:-4]) <= 990):
            label.append(1)
        else:
            label.append(0)
    return label
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers[:165]:
        layer.trainable = False
    for layer in base_model.layers[165:]:
        layer.trainable = True
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

FC_LAYERS = [512]
dropout = 0.5
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=2)
NUM_EPOCHS = 15
history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE,
                                       validation_data=valid_generator,
                        validation_steps=nb_validation_samples // BATCH_SIZE,
                                       shuffle=True)
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import efficientnet.keras as enet
from keras.backend import sigmoid

class SwishActivation(Activation):
    
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
get_custom_objects().update({'swish_act': SwishActivation(swish_act)})
model = enet.EfficientNetB0(include_top=False, input_shape=(224,224,3), pooling='avg', weights='imagenet')

# Adding 2 fully-connected layers to B0.
x = model.output

# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
predictions = layers.Dense(2, activation='sigmoid')(x)# Output layer

model_final = Model(inputs = model.input, outputs = predictions)


model_final.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])

num_train_images = 1317
nb_validation_samples = 329
NUM_EPOCHS = 10
BATCH_SIZE = 16
history = model_final.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE,
                                       validation_data=valid_generator,
                        validation_steps=nb_validation_samples // BATCH_SIZE,
                                       shuffle=True)



from fastai.imports import *
from fastai import *
from fastai.vision import *
from torchvision.models import *
#path_directory = '/kaggle/input/train_SOaYf6m/images/'
test=pd.read_csv('/kaggle/input/test_vc2kHdQ.csv')
train=pd.read_csv('/kaggle/input/train_SOaYf6m/train.csv')

path = '/kaggle/input/'
test_path = path+'Test Images/'
def genelabel(model, string):
    id_list=list(test.image_names)
    label=[]
    for iname in id_list:
        img=open_image(path+"train_SOaYf6m/images/"+iname)
        label.append(model.predict(img)[0])
        if (len(label)%350 == 0):
            print(f'{len(label)} images done!')
    return label

trfm =  get_transforms(do_flip = True, max_lighting = 0.2, max_zoom= 1.1, max_warp = 0.15, max_rotate = 45)
data = ImageDataBunch.from_csv(path, folder= 'train_SOaYf6m/images', 
                              valid_pct = 0.0,
                              csv_labels = 'train_SOaYf6m/train.csv',
                              ds_tfms = trfm, 
                              fn_col = 'image_names',
                              label_col = 'emergency_or_not',
                              bs = 16,
                              size = 300).normalize(imagenet_stats)

fbeta = FBeta(average='weighted', beta = 1)
learn = cnn_learner(data, models.resnet101, metrics=[accuracy, fbeta])
learn.fit(epochs = 30, lr = 1.5e-4)
import pandas as pd
sub = pd.read_csv('/kaggle/input/sample_submission_yxjOnvz.csv')
sub['resnet101'] = genelabel(learn, 'resnet101')
sub['resnet101'] = sub['resnet101'].astype('int')
sub.head()

fbeta = FBeta(average='weighted', beta = 1)
learn = cnn_learner(data, models.resnet50, metrics=[accuracy, fbeta])
learn.fit(epochs = 30, lr = 1.5e-4)
sub['resnet50'] = genelabel(learn, 'resnet50')
sub['resnet50'] = sub['resnet50'].astype('int')
sub.head()
del learn
learn = cnn_learner(data, models.densenet121, metrics=[accuracy, fbeta])
learn.fit(epochs = 50, lr = 6e-5)
sub['densenet121'] = genelabel(learn, 'densenet121')
sub['densenet121'] = sub['densenet121'].astype('int')
sub.head()
del learn
learn = cnn_learner(data, models.densenet161, metrics=[accuracy, fbeta])
learn.fit(epochs = 25, lr = 1e-4)
sub['densenet161'] = genelabel(learn, 'densenet161')
sub['densenet161'] = sub['densenet161'].astype('int')
sub.head()
sub.head()
new = sub.mode(axis='columns', numeric_only=True)
ans = new[0].tolist()

sub['emergency_or_not'] = ans
sub.head()
sub = sub.drop(['resnet101', 'densenet121', 'resnet50','densenet161'], axis = 1)
sub.head()
sub.to_csv('submitThis.csv',index=False)
