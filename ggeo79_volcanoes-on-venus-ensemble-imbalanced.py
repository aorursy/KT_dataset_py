# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from keras.layers import Input, Dense, Flatten, Average
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as prep_inputVgg16
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input as prep_inputResNet50
from keras.optimizers import Adam, RMSprop

sns.set()
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
np.set_printoptions(edgeitems=10)
TRAIN_PATH = '../input/volcanoesvenus/volcanoes_train/'
TEST_PATH = '../input/volcanoesvenus/volcanoes_test/'
IMAGE_HEIGHT_TARGET = 110
IMAGE_WIDTH_TARGET = 110
# Load train data
train_images = pd.read_csv(TRAIN_PATH + 'train_images.csv', header=None)
train_labels = pd.read_csv(TRAIN_PATH + 'train_labels.csv', header=None)
# Load test data
test_images = pd.read_csv(TEST_PATH + 'test_images.csv', header=None)
test_labels = pd.read_csv(TEST_PATH + 'test_labels.csv', header=None)
train_images.head()
train_labels.head()
train_labels = train_labels.drop([0])
nulls = []
for i in range(len(train_labels.index)):
    if (train_labels.iloc[i][0] == 1):
        nulls.append(train_labels.isnull().iloc[i][0])
# count nan values (true) in list
count_nans = sum(nulls)
count_nans
# Do the same for test labels
nulls = []
for i in range(len(test_labels.index)):
    if (test_labels.iloc[i][0] == 1):
        nulls.append(test_labels.isnull().iloc[i][0])
count_nans = sum(nulls)
count_nans
# Do the same for train and test images.
train_images.isnull().values.any()
test_images.isnull().values.any()
ax = sns.countplot(data = train_labels,x=train_labels[0][1:])
ax.set(xlabel='Volcanoes', ylabel='Count')
ax = sns.countplot(data = train_labels,x=train_labels[1][1:])
ax.set(xlabel='Type', ylabel='Count')
ax = sns.countplot(data = train_labels,x=train_labels[3][1:])
ax.set(xlabel='Number of volcanoes', ylabel='Count')
indices_train = np.where(train_labels.iloc[:, 0].astype(np.float) == 1)
train_labels.iloc[indices_train].shape
train_images.iloc[indices_train].shape
train_labels.head()
train_labels = train_labels.fillna('nan')
train_labels.iloc[:, 0] = (train_labels.iloc[:, 0]).str.replace('0', 'No')
train_labels.iloc[:, 0] = (train_labels.iloc[:, 0]).str.replace('1', 'Yes')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('1', 'Type 1')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('2', 'Type 2')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('3', 'Type 3')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('4', 'Type 4')
train_labels.iloc[:, 1] = (train_labels.iloc[:, 1]).str.replace('nan', 'Type nan')
train_labels.iloc[:, 3] = (train_labels.iloc[:, 3]).str.replace('nan', 'Nb volcanoes nan')
train_labels[:10]
labels = []
for idx in range(len(train_labels)):
    # index 0: Volcanoe or not
    # index 1: Type
    # index 3: Nb of volcanoes
    labels.append([train_labels.iloc[:, 0].values.item(idx), train_labels.iloc[:, 1].values.item(idx), train_labels.iloc[:, 3].values.item(idx)]) 
    
labels = np.array(labels)
#Show a few labels
labels[:4]
# Binarize the labels
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)
mlb.classes_
# Check the binarized labels
labels[:4]
def data():
    X_train, X_val, y_train, y_val  = train_test_split(train_images.values,
                                                       labels,
                                                       test_size=0.2,
                                                       stratify=labels,
                                                       random_state=1340)
        
    return X_train, X_val, y_train, y_val
X_train, X_val, y_train, y_val = data()
X_train_res = X_train.reshape((-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))
X_val_res = X_val.reshape((-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))
# Stack, in order to have 3 channels
X_train_vggnet = np.stack((np.squeeze(X_train_res),) * 3, -1)
X_val_vggnet = np.stack((np.squeeze(X_val_res),) * 3, -1)
# Preprocess input
X_train_vggnet = prep_inputVgg16(X_train_vggnet)
X_val_vggnet = prep_inputVgg16(X_val_vggnet)
train_data_gen = ImageDataGenerator(horizontal_flip=True,
                                    rotation_range=40,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2)
class VGGNet:
    @staticmethod
    def build(width, height, depth, classes, final_activ):
        # Initialize the model to use channels last
        input_shape = (height, width, depth)
        
        # In case where channels first is used
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            
        # Load pretrained weights
        imagenet_weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_vgg16 = VGG16(include_top=False, weights=imagenet_weights, input_shape=input_shape)
        last_layer = base_vgg16.output
        
        x = Flatten()(last_layer)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        preds_base_vgg16 = Dense(classes, activation=final_activ)(x)
        
        # Before compiling and train the model it is very important to freeze the convolutional base (resnet base).That means, preventing the weights from being updated during training.
        # If you omit this step, then the representations that were learned previously by the convolutional base will be modified during training.
        base_vgg16.trainable = False
        
        model_vgg16 = Model(base_vgg16.input, preds_base_vgg16)
               
        return model_vgg16
    
    def train(model, X, y, batch_size, epochs, class_weights, k_fold, loss, optimizer, metrics, model_checkpoint, early_stopping):
            
        # use k-fold cross validation test
        histories = []
        nb_validation_samples = len(X) // k_fold
        for fold in range(k_fold):
            x_training_data = np.concatenate([X[:nb_validation_samples * fold], X[nb_validation_samples * (fold + 1):]])
            y_training_data = np.concatenate([y[:nb_validation_samples * fold], y[nb_validation_samples * (fold + 1):]])

            x_validation_data = X[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]
            y_validation_data = y[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]

            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            history = model.fit_generator(train_data_gen.flow(x_training_data, y_training_data, batch_size=batch_size),
                                                              validation_data=[x_validation_data, y_validation_data],
                                                              epochs = epochs,
                                                              shuffle=True,
                                                              verbose=2,
                                                              class_weight=class_weights,
                                                              steps_per_epoch = int(len(X_train) / batch_size),
                                                              validation_steps =int(len(X_val) / batch_size),
                                                              callbacks=[model_checkpoint, early_stopping])
            histories.append(history)
        
        return histories, model
model_vgg16 = VGGNet.build(IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 3, len(mlb.classes_), 'sigmoid')
final_activation = 'sigmoid'
batch_size = 32
epochs = 100
k_fold = 3
loss = 'binary_crossentropy'
adam = Adam(lr=0.0001)
optimizer = adam
metrics = ['accuracy']
early_stopping = EarlyStopping(patience=10, verbose=1)
model_vgg16_checkpoint = ModelCheckpoint('./b_32_relu_optim_adam.hdf5', verbose=1, save_best_only=True)
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[:, -1]), y_train[:, -1]) #-1 is the lats index (volcanoe or not)
# Concatenate train and test data
X_data = np.concatenate((X_train_vggnet, X_val_vggnet))
y_data = np.concatenate((y_train, y_val))
history_vgg16, model_vgg16 = VGGNet.train(model_vgg16,
                                          X_data,
                                          y_data,
                                          batch_size,
                                          epochs,
                                          class_weights,
                                          k_fold,
                                          loss,
                                          optimizer,
                                          metrics,
                                          model_vgg16_checkpoint,
                                          early_stopping)
fig, axes = plt.subplots(k_fold, 2, figsize=(20, 12))

for i in range(k_fold):
    
    axes[i, 0].plot(history_vgg16[i].epoch, history_vgg16[i].history['loss'], label='Train loss')
    axes[i, 0].plot(history_vgg16[i].epoch, history_vgg16[i].history['val_loss'], label='Val loss')
    axes[i, 0].legend()

    axes[i, 1].plot(history_vgg16[i].epoch, history_vgg16[i].history['acc'], label = 'Train acc')
    axes[i, 1].plot(history_vgg16[i].epoch, history_vgg16[i].history['val_acc'], label = 'Val acc')
    axes[i, 1].legend()

 
plt.tight_layout()
class ResNet:
    @staticmethod
    def build(width, height, depth, classes, final_activ):
        # Initialize the model to use channels last
        input_shape = (height, width, depth)
        
        # In case where channels first is used
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            
        # Load pretrained weights
        imagenet_weights = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        base_resnet = ResNet50(include_top=False, weights=imagenet_weights, input_shape=input_shape)
        last_layer = base_resnet.output
        
        x = Flatten()(last_layer)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        preds_base_resnet = Dense(classes, activation=final_activ)(x)
        
        # Before compiling and train the model it is very important to freeze the convolutional base (resnet base).That means, preventing the weights from being updated during training.
        # If you omit this step, then the representations that were learned previously by the convolutional base will be modified during training.
        base_resnet.trainable = False
        
        model_resnet = Model(base_resnet.input, preds_base_resnet)
               
        return model_resnet
    
    def train(model, X, y, batch_size, epochs, class_weights, k_fold, loss, optimizer, metrics, model_checkpoint, early_stopping):
        
        # use k-fold cross validation test
        histories = []
        nb_validation_samples = len(X) // k_fold
        for fold in range(k_fold):
            x_training_data = np.concatenate([X[:nb_validation_samples * fold], X[nb_validation_samples * (fold + 1):]])
            y_training_data = np.concatenate([y[:nb_validation_samples * fold], y[nb_validation_samples * (fold + 1):]])
            
            x_validation_data = X[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]
            y_validation_data = y[nb_validation_samples * fold:nb_validation_samples * (fold + 1)]
            
            model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            
            history = model.fit_generator(train_data_gen.flow(x_training_data, y_training_data, batch_size=batch_size),
                                                              validation_data=[x_validation_data, y_validation_data],
                                                              epochs = epochs,
                                                              shuffle=True,
                                                              verbose=2,
                                                              class_weight=class_weights,
                                                              steps_per_epoch = int(len(X_train) / batch_size),
                                                              validation_steps =int(len(X_val) / batch_size),
                                                              callbacks=[model_checkpoint, early_stopping])
            histories.append(history)
            
        return histories, model
model_resnet = ResNet.build(IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 3, len(mlb.classes_), 'sigmoid')
X_train_resnet = np.stack((np.squeeze(X_train_res),) * 3, -1)
X_val_resnet = np.stack((np.squeeze(X_val_res),) * 3, -1)
X_train_resnet = prep_inputResNet50(X_train_resnet)
X_val_resnet = prep_inputResNet50(X_val_resnet)
# Concatenate train and test data
X_data_resnet = np.concatenate((X_train_resnet, X_val_resnet))
y_data_resnet = np.concatenate((y_train, y_val))
model_checkpoint_resnet = ModelCheckpoint('./b_32_relu_optim_adam_resnet.hdf5', verbose=1, save_best_only=True)
history_resnet, model_resnet = ResNet.train(model_resnet,
                                            X_data_resnet,
                                            y_data_resnet,
                                            batch_size,
                                            epochs,
                                            class_weights,
                                            k_fold,
                                            loss,
                                            optimizer,
                                            metrics,
                                            model_checkpoint_resnet,
                                            early_stopping)
fig, axes = plt.subplots(k_fold, 2, figsize=(20, 12))

for i in range(k_fold):
    
    axes[i, 0].plot(history_resnet[i].epoch, history_resnet[i].history['loss'], label='Train loss')
    axes[i, 0].plot(history_resnet[i].epoch, history_resnet[i].history['val_loss'], label='Val loss')
    axes[i, 0].legend()

    axes[i, 1].plot(history_resnet[i].epoch, history_resnet[i].history['acc'], label = 'Train acc')
    axes[i, 1].plot(history_resnet[i].epoch, history_resnet[i].history['val_acc'], label = 'Val acc')
    axes[i, 1].legend()

 
plt.tight_layout()
# Load the best saved model
#model_vgg16.load_weights(filepath='../input/volcanoes-on-venus-ensemble-imbalanced/b_32_relu_optim_adam.hdf5')
#model_resnet.load_weights(filepath='../input/volcanoes-on-venus-ensemble-imbalanced/b_32_relu_optim_adam_resnet.hdf5')
def ensemble(models):
    input_image = Input(shape=(IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 3))
    
    vgg16_out = models[0](input_image)
    resnet_out = models[1](input_image)

    output = Average()([vgg16_out, resnet_out])
    model = Model(input_image, output)
    
    return model
# Combine all models
models = [model_vgg16, model_resnet]
ensemble_model = ensemble(models)
test_images.head()
test_labels.head()
test_labels = test_labels.drop([0])
# Replace all float nans with string nans.
test_labels = test_labels.fillna('nan')
# Replace 0 or 1 with No or Yes
test_labels.iloc[:, 0] = (test_labels.iloc[:, 0]).str.replace('0', 'No')
test_labels.iloc[:, 0] = (test_labels.iloc[:, 0]).str.replace('1', 'Yes')
# Replace 1,2,3,4 with Type1,2,3,4
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('1', 'Type 1')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('2', 'Type 2')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('3', 'Type 3')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('4', 'Type 4')
test_labels.iloc[:, 1] = (test_labels.iloc[:, 1]).str.replace('nan', 'Type nan')
test_labels.iloc[:, 3] = (test_labels.iloc[:, 3]).str.replace('nan', 'Nb volcanoes nan')
tmp = train_labels.iloc[:, 3].values
idx, = np.where(tmp != 'Nb volcanoes nan')
idx_greater = idx[tmp[idx].astype(int) > 3]
idx_greater
series_list_images = [pd.Series(train_images.iloc[425, :], index=test_images.columns ) ,
                      pd.Series(train_images.iloc[1513, :], index=test_images.columns )]

series_list_labels = [pd.Series(train_labels.iloc[425, :], index=test_labels.columns ) ,
                      pd.Series(train_labels.iloc[1513, :], index=test_labels.columns )]

test_images_full = test_images.append(series_list_images , ignore_index=True)
test_labels_full = test_labels.append(series_list_labels , ignore_index=True)
labels_test = []
for idx in range(len(test_labels_full)):
    # index 0: Volcanoe or not
    # index 1: Type
    # index 3: Nb of volcanoes
    labels_test.append([test_labels_full.iloc[:, 0].values.item(idx), test_labels_full.iloc[:, 1].values.item(idx), test_labels_full.iloc[:, 3].values.item(idx)]) 
    
labels_test = np.array(labels_test)
# Binarize the labels
labels_test = mlb.fit_transform(labels_test)
# Check classes
mlb.classes_
X_test = test_images_full
y_test = labels_test
# Reshape test data and create 3 channels
X_test = X_test.values.reshape((-1, IMAGE_HEIGHT_TARGET, IMAGE_WIDTH_TARGET, 1))
X_test = np.stack((np.squeeze(X_test),) * 3, -1)
# Preprocess data
X_test = prep_inputVgg16(X_test)
# predict on validation and test data
pred_val = ensemble_model.predict(X_val_vggnet) 
pred_test = ensemble_model.predict(X_test, batch_size=batch_size)
# Squeeze one dimension to be able to plot
X_train_squeeze = X_train_vggnet.squeeze()
y_train_squeeze = y_train.squeeze()
pred_val_squeeze = pred_val.squeeze()
X_val_squeeze = X_val_vggnet.squeeze()
y_val_squeeze = y_val.squeeze()
X_test_squeeze = X_test.squeeze()
def scale_image(input_data, min_orig, max_orig, min_target, max_target):
    orig_range = max_orig - min_orig
    target_range = max_target - min_target
    scaled_data = np.array((input_data - min_orig) / float(orig_range))
    return min_target + (scaled_data * target_range)
X_test_denorm = scale_image(X_test_squeeze, X_test_squeeze.min(), X_test_squeeze.max(), 0, 1)
plt.rc('text', usetex=False)
max_images = 6

fig, axes = plt.subplots(max_images//2, 2, figsize=(22, 18))
axes = axes.ravel()

idxlist = [0, 1, 2, 3, 4, 5]
for i in  range(max_images):   

    #idx = np.random.randint(0, len(X_test)-1)
    idx = idxlist[i]
    
    axes[i].grid(False)
    axes[i].imshow(X_test_denorm[idx], cmap='Greens')
    axes[i].set_title(mlb.inverse_transform(y_test[idx:idx+1, :]))
    
    label_img = []
    for (label, p) in zip(mlb.classes_, pred_test[idx]):
        #label_img.append("{0}: {1}%".format(label.astype(np.str), (p * 100).astype(np.float32) ))
        label_img.append((p * 100).astype(np.float32))
        
    text = 'Volcanoe\nYes: {0:.2f}% No: {1:.2f}%\nType\n1: {2:.2f}% 2: {3:.2f}% 3: {4:.2f}% 4: {5:.2f}% nan: {6:.2f}%\n\
            Nb.Volcanoes\n1: {7:.2f}% 2: {8:.2f}% 3: {9:.2f}% 4: {10:.2f}% 5: {11:.2f}% nan: {12:.2f}%'\
            .format(label_img[12], label_img[6], label_img[7], label_img[8], label_img[9], label_img[10], label_img[11], label_img[0], label_img[1],\
                   label_img[2], label_img[3], label_img[4], label_img[5])
    
    axes[i].text(55, 95, text , size=12, ha="center", va="center",
            bbox=dict( fc=(1., 1., 0.8), alpha=0.7))
    
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout()
class Regression:
    @staticmethod
    def build(X):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_shape=(X.shape[1],)))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1))
       
        return model
    
    def train(X, y, k, optimizer, batch_size, epochs):
                    
        # use k-fold cross validation test
        kfold = KFold(n_splits=k, shuffle=True, random_state=1340)
        histories = []
        for train, test in kfold.split(X):
            model = Regression.build(X)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
            
            history = model.fit(X[train],
                                y[train],
                                validation_data=[X[test], y[test]],
                                batch_size=batch_size,
                                epochs = epochs,
                                verbose=0)
            
            # evaluate the model
            #val_mse, val_mae = model.evaluate(X[test], y[test], verbose=0)
            mae_history = history.history['val_mean_absolute_error']
            histories.append(mae_history)
                    
        return histories, model
indices_radius, = np.where(train_labels.iloc[:, 0] == 'Yes')
X_volcanoes_radius = train_images.iloc[indices_radius]
# Volcanoes labels
y_volcanoes_radius = train_labels.iloc[indices_radius]
# take only the radius
y_volcanoes_radius = y_volcanoes_radius.iloc[:, 2]
def data_radius():
    X_train, X_val, y_train, y_val  = train_test_split(X_volcanoes_radius,
                                                       y_volcanoes_radius.values.astype(np.float32).reshape(-1, 1),
                                                       test_size=0.2,
                                                       random_state=1340)
        
    return X_train, X_val, y_train, y_val
X_train_radius, X_val_radius, y_train_radius, y_val_radius = data_radius()
# Concatenate train and test data
X_data_radius = np.concatenate((X_train_radius, X_val_radius))
y_data_radius = np.concatenate((y_train_radius, y_val_radius))
# initialize scaler
scaler = StandardScaler()
# scale data
scaler.fit(X_data_radius)
X_std_radius = scaler.transform(X_data_radius)
batch_size_reg = 16
epochs = 200
k = 3
rmsprop = RMSprop(lr=0.001)

history_mae, model_reg = Regression.train(X_std_radius, y_data_radius, k, rmsprop, batch_size_reg, epochs)
y_data_check = y_data_radius.astype(np.float32)
# Find out max radius value
np.amax(y_data_check)
# Find out min radius value
np.amin(y_data_check)
# Let's plot the mae
avg_mae = [np.mean([x[i] for x in history_mae]) for i in range(epochs)]

plt.plot(range(1, len(avg_mae) + 1), avg_mae)
plt.xlabel('Epochs')
plt.ylabel('Val mae')
plt.show()
indices_test,  = np.where(pred_test[:, 12] >= 0.5)
# Volcanoes test images
X_volcanoes_test = test_images_full.iloc[indices_test]
# scale test data
X_volcanoes_test_std = scaler.transform(X_volcanoes_test.values.astype(np.float32))
# Volcanoes test labels
y_volcanoes_test = test_labels_full.iloc[indices_test]
# take only the radius
y_volcanoes_test = y_volcanoes_test.iloc[:, 2]
y_volcanoes_test = pd.to_numeric(y_volcanoes_test, errors='coerce')
y_volcanoes_test = y_volcanoes_test.fillna(0)
test_mse, test_mae = model_reg.evaluate(X_volcanoes_test_std, y_volcanoes_test.values.astype(np.float32))
test_mse
test_mae
pred_regression = model_reg.predict(X_volcanoes_test_std)
# Check a few values
for i in range(100):
    idx = np.random.randint(0, len(y_volcanoes_test.values)-1)
    print('Real: {0}\t Pred: {1:.2f}'.format(y_volcanoes_test.values[idx], pred_regression.squeeze()[idx]))