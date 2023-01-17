# Load packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from random import shuffle
import os, gc, time, cv2, random, math

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

####################
# Global Constants #
####################
INCEPTION_V3_WEIGHTS_PATH = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
PATH = '../input/dogs-vs-cats-redux-kernels-edition/'
TRAIN_DIR = PATH+'train'
TEST_DIR =  PATH+'test'
NUM_CLASSES = 2
IMG_SIZE = 145  ###
CHANNELS = 3
EPOCHS = 30
BATCH_SIZE = 32

train_images = os.listdir(TRAIN_DIR)
test_images = os.listdir(TEST_DIR)

# # For testing purposes
# train_images = train_images[:10000]
# test_images = test_images[:100]
def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat': return 0  ###
    elif word_label == 'dog' : return 1  ###

# Return a numpy array of train and test data
def process_data(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = label_img(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img), label])
    shuffle(data_df)
    return data_df
# Prepare the train data
train_data = process_data(train_images, TRAIN_DIR, isTrain=True)
X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array([i[1] for i in train_data])
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras import optimizers
from keras.layers import Conv2D, Dense, Input, Flatten, Concatenate, Dropout, Activation
from keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from keras import applications

def get_pretrained_model(Weights_path='imagenet', trainable=False, Input_Shape=None):
    input_shape = Input_Shape
    base_model = InceptionV3(weights=None, include_top=False, input_shape= input_shape)
    base_model.load_weights(Weights_path)
    for l in base_model.layers:
        l.trainable = trainable
    return base_model
    
def build_model(PreModel, LearningRate=1e-3, Decay=1e-8):
    
    input_x = PreModel.inputs
    
    x_model = PreModel.output
    #x_model = GlobalAveragePooling2D()(x_model)
    
    x_model = Flatten()(x_model)
    
    x_model = Dense(64, activation='relu',name='fc1_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_1')(x_model)
    x_model = BatchNormalization()(x_model)
    
    x_model = Dense(32, activation='relu',name='fc2_Dense')(x_model)
    x_model = Dropout(0.5, name='dropout_2')(x_model)
    x_model = BatchNormalization()(x_model)
    
    predictions = Dense(1, activation='sigmoid',name='output_layer')(x_model)
    model = Model(inputs=input_x, outputs=predictions)
    optimizer = optimizers.SGD(lr=LearningRate, decay=Decay)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

PreModel = get_pretrained_model(Weights_path=INCEPTION_V3_WEIGHTS_PATH,
                                trainable=False,
                                Input_Shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
model = build_model(PreModel, LearningRate=1e-3, Decay=1e-2)
# Model Summary
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

# model.summary()
# plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))
# Trainable layers
for l in model.layers:
    if l.trainable: print(l.name)
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=1)

# Augmentation configuration to use for training and validation
train_datagen = ImageDataGenerator(
        rescale=1./255,#!!!!!
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20, 
        horizontal_flip=True,
#         preprocessing_function=preprocess_input
)
test_datagen = ImageDataGenerator(
    rescale=1./255,#!!!!!
#     preprocessing_function=preprocess_input
)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
BestModelWeightsPath = 'BestModel.hdf5'
check_point = ModelCheckpoint(
    BestModelWeightsPath, monitor='val_loss', verbose=1,
    save_best_only=True, 
    save_weights_only=True,
    mode='min'
)
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=3, verbose=1)
earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=30)
callbacks_list = [check_point, lr_reduce, earlyStop]

K.set_value(model.optimizer.lr, 0.1) ####
gc.collect()
history = model.fit_generator(
    train_datagen.flow(np.array(X_train), y_train, batch_size=BATCH_SIZE, shuffle=True),
    steps_per_epoch= len(X) // BATCH_SIZE,
    validation_data = test_datagen.flow(np.array(X_val), y_val, batch_size=BATCH_SIZE*3, shuffle=False),
    validation_steps = len(X_val) // (BATCH_SIZE*3),
    epochs=EPOCHS,
    shuffle=False,
    verbose=1,
    callbacks=callbacks_list
)
# Plotting loss and accuracy for the model
def plot_accuracy_and_loss(history):
    eval_res = pd.DataFrame(history.history)
    f, ax = plt.subplots(1,2, figsize=(18,5))
    for i, c in enumerate(['acc', 'loss']):
        ax[i].plot(eval_res[[c]], label=f'Training {c}')
        ax[i].plot(eval_res[[f'val_{c}']], label=f'Validation {c}')
        ax[i].set_xlabel('Epoch'); ax[i].set_ylabel(c);
        ax[i].legend(); ax[i].set_title(f'Training and validation {c}'); plt.grid();
    plt.show()
plot_accuracy_and_loss(history)
last_5_layer_names = [_.name for _ in PreModel.layers[::-1][:5]]
print(f'Pretrained have {len(PreModel.layers)} layers')
print(f'My model have {len(model.layers)} layers')
print(f'Pretrained last 5 layers: ', last_5_layer_names, '\n')

# for l in model.layers[:]: # enable training just for all layers
for l in model.layers[::-1][6:12]: # enable training just for last five layers of the Restnet50
    print('Fine-tune', l.name);
    l.trainable = True
BestModelWeightsPath = 'BestModel.hdf5'
check_point = ModelCheckpoint(
    BestModelWeightsPath, monitor='val_loss', verbose=1,
    save_best_only=True, 
    save_weights_only=True,
    mode='min'
)
lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=3, verbose=1)
earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=30)
callbacks_list = [check_point, lr_reduce, earlyStop]

K.set_value(model.optimizer.lr, 1e-5) ###
K.set_value(model.optimizer.decay, 1e-8)
gc.collect()
history = model.fit_generator(
    train_datagen.flow(np.array(X_train), y_train, batch_size=BATCH_SIZE, shuffle=True),
    steps_per_epoch= len(X) // BATCH_SIZE,
    validation_data = test_datagen.flow(np.array(X_val), y_val, batch_size=BATCH_SIZE*3, shuffle=False),
    validation_steps = len(X_val) // (BATCH_SIZE*3),
    epochs=math.ceil(EPOCHS*1.6), ###
    verbose=1,
    callbacks=callbacks_list
)
plot_accuracy_and_loss(history)
# Free some memory
del X, y, train_data; gc.collect()

# Load Best model weights
model.load_weights(BestModelWeightsPath)

# Testing Model on Test Data
test_data = process_data(test_images, TEST_DIR, isTrain=False)
f, ax = plt.subplots(5,5, figsize=(18,18))
for i,data in enumerate(test_data[:25]):
    img_num = data[1]
    img_data = data[0]
    orig = img_data
    data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
    data = data* 1./255
    model_out = model.predict([data])[0]
    if model_out[0] >= 0.5: 
        str_predicted='Dog'
    else: 
        str_predicted='Cat'
    ax[i//5, i%5].imshow(orig)
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_title("Confident :{:.2%} as {} ".format(abs(0.5-model_out[0])*2, str_predicted))    
plt.show()
prob = []
img_list = []
for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(-1,IMG_SIZE,IMG_SIZE,3)
        data = data* 1./255
        model_out = model.predict([data])[0]
        img_list.append(img_num)
        prob.append(model_out[0])
    
submission = pd.DataFrame({'id':img_list , 'label':prob})
print(submission.head())
submission.to_csv("submit.csv", index=False)
