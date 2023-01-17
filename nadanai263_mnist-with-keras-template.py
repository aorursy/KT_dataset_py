%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import glorot_uniform as Xavier
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
import os
import itertools
print('Tensorflow version', tf.__version__)
print('Keras version', keras.__version__)
print(os.listdir("../input/"))
DIR = '../input/digit-recognizer/'
DIR_IN = '../input/mnist-with-keras-template/'
DIR_OUT = './'
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 50
# Define useful functions

# Real-time plot updates for Keras (credits: https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e)
class PlotLearn(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        
        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), sharex=True)
                
        clear_output(wait=True)

        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss');
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy');
        ax2.legend()

        plt.show(); 
        
plot_learn = PlotLearn()

# Save Keras model
def savemodel(model, model_params_dir,name):
    json_string = model.to_json()
    # Save model architecture in JSON file
    open(model_params_dir + name + '_arch.json', 'w').write(json_string)
    # Save weights as HDF5
    model.save_weights(model_params_dir + name + '_weights.h5')
    print("Saved model to disk")

# Load Keras model
def loadmodel(model_params_dir,name): 
    # Load model architecture from JSON file
    model = keras.models.model_from_json(open(model_params_dir + name + '_arch.json').read())
    # Load model weights from HDF5 file
    model.load_weights(model_params_dir + name + '_weights.h5')
    print("Loaded model from disk")
    return model

# Load weights only
def loadmodelweights(model,model_params_dir,name_weightfile): 
    # Load model weights from HDF5 file
    model.load_weights(model_params_dir + name_weightfile+'.h5')
    print("Loaded model weights from disk")
    return model
# Load all data into memory
df_train=pd.read_csv(DIR + 'train.csv')
df_test=pd.read_csv(DIR + 'test.csv')
# Convert images
images_train=np.asarray(df_train.drop(['label'],axis=1,inplace=False))
labels_train=np.asarray(df_train['label'])
images_test=np.asarray(df_test)

# input image dimensions
img_rows, img_cols = 28, 28

# Pre-process input data
x_train, y_train, x_test = images_train,labels_train, images_test

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') # Reshape and convert fmt
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32')
x_train /= 255
x_test /= 255

input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('input shape:', input_shape)
print(x_train.shape[0], 'Number of training samples')
print(x_test.shape[0], 'Number of test samples')
# convert class vectors from digit labels to binary class matrices (N x 10)
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_train.shape
# Shuffle data and generate dev set

percent_holdout=0.02 # percentage of data to hold out

# Set seed for reproducible results
np.random.seed(1)

# Select random subset by index
inds=np.arange(y_train.shape[0])
np.random.shuffle(inds) # Shuffle indices in-place

# Carve into train/dev/test sets
dev_inds=inds[int((1-percent_holdout)*len(inds)):] # Dev set
train_inds=inds[:int((1-percent_holdout)*len(inds))] # Training set 

# Look at final distributions
plt.hist(train_inds); 
plt.hist(dev_inds); plt.show()

print('Number of training examples', len(train_inds))
print('Number of dev examples', len(dev_inds))

x_train_subset=x_train[train_inds]
y_train_subset=y_train[train_inds]
x_dev=x_train[dev_inds]
y_dev=y_train[dev_inds]
# Define model graph

model_name = 'CNN1'

x_input = Input(shape=(28,28,1), name='Input1')
x = x_input

# Convs. BNs after Relus!

x = Conv2D(32,(3, 3), padding='same', name='C1_1')(x)
x = Activation('relu', name='Relu1_1')(x)
x = BatchNormalization(axis=-1, name='BN1_1')(x)

x = Conv2D(32,(3, 3), padding='same', name='C1_2')(x)
x = Activation('relu', name='Relu1_2')(x)
x = BatchNormalization(axis=-1, name='BN1_2')(x)

x = Conv2D(32,(5, 5), padding='same', name='C1_3')(x)
x = Activation('relu', name='Relu1_3')(x)
x = BatchNormalization(axis=-1, name='BN1_3')(x)

x = MaxPooling2D(pool_size=(2, 2), name='MP1')(x)
x = Dropout(0.4, name='Drop1')(x)


x = Conv2D(64,(3, 3), padding='same', name='C2_1')(x)
x = Activation('relu', name='Relu2_1')(x)
x = BatchNormalization(axis=-1, name='BN2_1')(x)

x = Conv2D(64,(3, 3), padding='same', name='C2_2')(x)
x = Activation('relu', name='Relu2_2')(x)
x = BatchNormalization(axis=-1, name='BN2_2')(x)

x = Conv2D(64,(5, 5), padding='same', name='C2_3')(x)
x = Activation('relu', name='Relu2_3')(x)
x = BatchNormalization(axis=-1, name='BN2_3')(x)

x = MaxPooling2D(pool_size=(2, 2), name='MP2')(x)
x = Dropout(0.4, name='Drop2')(x)


x = Conv2D(128,(3, 3), padding='same', name='C3')(x)
x = Activation('relu', name='Relu3')(x)
x = BatchNormalization(axis=-1, name='BN3')(x)
x = MaxPooling2D(pool_size=(2, 2), name='MP3')(x)
xinter = Flatten(name='Features')(x)

x = Dropout(0.5, name='Drop4')(xinter)
x = Dense(NUM_CLASSES, name='D4')(x)
x = Activation('softmax', name='Output')(x)

model = Model(inputs=x_input, outputs=x)
model_features_only = Model(inputs=x_input, outputs=xinter)

model.summary()
# Compile
adam=Adam(lr=0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=adam,
              metrics=['accuracy'])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=2, 
                                            factor=0.5, 
                                            min_lr=0.000001)

# Create checkpoint: save model after every epoch
modelcheckpoint = keras.callbacks.ModelCheckpoint(DIR_OUT + model_name +'_bestweights.h5',
                                              monitor = 'val_loss',
                                              verbose = True,
                                              save_best_only = True,
                                              mode = 'auto')
# Run!
history=model.fit(x_train_subset, y_train_subset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=(x_dev, y_dev),
            callbacks=[modelcheckpoint,learning_rate_reduction])

# Print evaluation
score = model.evaluate(x_train_subset,y_train_subset,verbose=0)
print('Train loss:', score[0])
print('Train Accuracy:', score[1])

score = model.evaluate(x_dev,y_dev,verbose=0)
print('Dev loss:', score[0])
print('Dev Accuracy:', score[1])

# save the model 
savemodel(model, DIR_OUT, model_name)
print('Completed ', len(history.epoch), 'epochs.')
# Plot train history
fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
ax[1].set_title('accuracy')
ax[1].plot(history.epoch, history.history["acc"], label="Train accuracy")
ax[1].plot(history.epoch, history.history["val_acc"], label="Validation accuracy")
ax[0].legend()
ax[1].legend()
plt.savefig(DIR_OUT+model_name+'_trainprofile.pdf',bbox_inches='tight')
# Load a model if you want:
#model=loadmodel(DIR_OUT,'CNN1')
# or take best weights:
#model=loadmodelweights(model,DIR_IN,model_name+'_bestweights') # Load weights only
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=adam,metrics=['accuracy'])
# Look at confusion matrix 

model=loadmodelweights(model,DIR_IN,model_name+'_bestweights') # Load best weights only

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(x_dev)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_dev,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Create submission file
Y_pred = model.predict(x_test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
ids = np.arange(Y_pred_classes.shape[0])+1

data = {'ImageId':ids, 'Label':Y_pred_classes}
pred=pd.DataFrame(data)
pred.to_csv(DIR_OUT+'preds1.csv',sep=',',index=False)
