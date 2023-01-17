import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Input
from keras import applications, optimizers
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_data_dir = "../input/myautoge-cars-data/training_set/training_set"
input_shape = (128,128)
batch_size = 128
train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    vertical_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
) # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
) # set as validation data


from keras import backend as K
import keras

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

METRICS = [
      precision_m,
      recall_m,
      f1_m,
      keras.metrics.AUC(name='auc', num_thresholds=200, curve='ROC', summation_method='interpolation',
            dtype=None, thresholds=None, multi_label=False, label_weights=None),
]


cnn = Sequential()

cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128, 3)))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(5, activation='softmax'))
cnn.summary()
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=METRICS)
# # Generate a model with all layers (with top)
# vgg16 = VGG16(weights=None,include_top=True, input_shape=(64, 64, 3))

# #Add a layer where input is the output of the  second last layer 
# x = Dense(5, activation='softmax', name='predictions')(vgg16.layers[-2].output)

# #Then create the corresponding model 
# cnn = Model(vgg16.input, x)
# cnn.summary()
pat = 5 #this is the number of epochs with no improvment after which the training will stop
early_stopping = EarlyStopping(monitor='val_loss', patience=pat, verbose=1)

#define the model checkpoint callback -> this will keep on saving the model as a physical file
model_checkpoint = ModelCheckpoint('result_model.h5', verbose=1, save_best_only=True)

history = cnn.fit(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            validation_data = validation_generator, 
            validation_steps = validation_generator.samples // batch_size,
            epochs = 200,
            callbacks=[model_checkpoint]
)
validation_generator.samples
Y_pred = cnn.predict_generator(validation_generator, validation_generator.samples // batch_size+1, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(validation_generator.classes, y_pred)
print(cm)
print('Classification Report')
target_names = ["Ford", "Hyundai", "Lexus", "Mercedes-benz", "Toyota"]
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'f1' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'f1' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training f1-score (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation f1-score (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('F1 score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 score')
    plt.legend()
    plt.show()
    
plot_history(history)
def get_model():
    cnn = Sequential()

    cnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128,128, 3)))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(256, activation='relu'))
    cnn.add(Dense(256, activation='relu'))
    cnn.add(Dense(5, activation='softmax'))
    
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
    
    return cnn
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.externals.joblib import parallel_backend
# from time import time


# start=time()

# optimizer = ['rmsprop', 'adam']
# epochs = [50, 100, 200, 300]
# batches = [64, 128, 256]


# model = KerasClassifier(build_fn=get_model, verbose=0)
# param_grid = dict(nb_epoch=epochs, batch_size=batches)

# grid = GridSearchCV(estimator=model, 
#                     param_grid=param_grid,
#                     scoring = 'accuracy',
#                     cv = 10,
#                     verbose=10)

# X, y = train_generator.next()
# y = np.argmax(y, axis=1)

# grid_result = grid.fit(X, y)

# print("----------------Done---------------")

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# print("total time:",time()-start)