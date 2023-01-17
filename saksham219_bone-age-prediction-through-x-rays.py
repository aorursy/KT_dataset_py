import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from PIL import Image
print(os.listdir("../input"))
from tensorflow.python import keras
df = pd.read_csv('../input/boneage-training-dataset.csv')
df.shape
df.head(15)
training_images_path = '../input/boneage-training-dataset/boneage-training-dataset/'
#os.listdir(training_images_path)
#os.path.join(training_images_path, '1377.png')

random_numbers = np.random.randint(1,df.shape[0],5)
fig, ax = plt.subplots(1,5, figsize = (20,20))
for i,n in enumerate(random_numbers):
    img = Image.open(os.path.join(training_images_path, str(df.ix[n,'id']) +'.png'))
    ax[i].imshow(np.array(img))
    ax[i].set_title('age: {}'.format(df.ix[n,'boneage']))
df[['boneage','male']].hist(figsize = (10,5))
boneage_std = 2 * df['boneage'].std()
boneage_mean = df['boneage'].mean()
df['norm_age'] = (df['boneage'] - boneage_mean)/boneage_std
df.head()
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size = 0.20, random_state = 21)
print(train_df.shape)
print(val_df.shape)
from keras.preprocessing.image import ImageDataGenerator
img_size = (96,96)
img_rows = 96
img_cols = 96

image_paths_train = train_df['id'].apply(lambda x: os.path.join(training_images_path, str(x)) + '.png' ).values
image_paths_val = val_df['id'].apply(lambda x:os.path.join(training_images_path, str(x)) + '.png').values

#color_mode = 'grayscale' 
#class_mode = 'sparse' --> for regression problems

data_generator = ImageDataGenerator(samplewise_center = True, samplewise_std_normalization = True, horizontal_flip = True, width_shift_range = 0.2, height_shift_range = 0.2)
train_generator = data_generator.flow_from_directory(training_images_path, target_size = img_size, batch_size = 64,  class_mode = 'sparse', color_mode = 'grayscale', shuffle = False)
train_generator.filenames = image_paths_train
train_generator.classes =  train_df['norm_age'].values
train_generator.samples = train_df.shape[0]
train_generator.n = train_df.shape[0]
train_generator.directory = ''
train_generator._set_index_array()


data_generator_no_aug =  ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
valid_generator = data_generator_no_aug.flow_from_directory(training_images_path, target_size =img_size, batch_size = 256, 
                                                            class_mode = 'sparse', color_mode = 'grayscale', shuffle = False)
valid_generator.filenames = image_paths_val
valid_generator.classes = val_df['norm_age'].values
valid_generator.samples = val_df.shape[0]
valid_generator.n = val_df.shape[0]
valid_generator.directory = ''
valid_generator._set_index_array()

fixed_val = data_generator_no_aug.flow_from_directory(training_images_path, target_size =img_size, batch_size = 2048, 
                                                      class_mode = 'sparse', color_mode = 'grayscale', shuffle = False)
fixed_val.filenames = image_paths_val
fixed_val.classes = val_df['norm_age'].values
fixed_val.samples = val_df.shape[0]
fixed_val.n = val_df.shape[0]
fixed_val.directory = ''
fixed_val._set_index_array()
test_x, test_y = next(fixed_val)
test_x.shape, test_y.shape
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
model = Sequential()
model.add(BatchNormalization(input_shape = (img_rows, img_cols,1)))
model.add(Conv2D(filters = 32, kernel_size = (2,2), strides = 2, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 32, kernel_size = (2,2), strides = 2, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 32, kernel_size = (2,2), activation = 'relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(1, activation = 'linear'))
model.summary()
from keras.metrics import mean_absolute_error
def mae(preds, true_preds):
    return mean_absolute_error(preds * boneage_std + boneage_mean, true_preds * boneage_std + boneage_mean)

adam = keras.optimizers.Adam(lr = 0.001, decay = 0)
sgd = keras.optimizers.SGD(lr = 0.1, momentum = 0.9)
model.compile(optimizer = adam , loss = 'mse' , metrics = [mae])
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weights_path = 'best_weights1.hd5'
checkpoint = ModelCheckpoint(weights_path, monitor = 'val_loss', save_best_only = True, save_weights_only = True, mode = 'min', verbose =1)
import tensorflow as tf
from tensorflow.python.keras import backend as K

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lrs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        lr = self.model.optimizer.lr
        self.lrs.append(K.cast(lr, 'float32'))
        
    def on_train_end(self, logs = None):
        plt.plot(np.arange(0,len(self.losses),1),self.losses, color = 'blue')
        plt.title('Loss with batches')
        plt.xlabel('batch')
        plt.ylabel('mse loss')
        plt.show()

        
    '''def on_epoch_end(self, epoch, logs = {})
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * Keras.cast(iterations, K.dtype(decay)))
        self.lrs.append(lr_with_decay)'''
    
class LRFinder(Callback):
    def __init__(self, max_batches = 5000, base_lr = 1e-4, max_lr = 0.1, lr_step_size = 1e-4):
        self.max_batches = max_batches
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.lr_step_size = lr_step_size
        self.lr = 0
        self.lrs = []
        self.losses = []
        self.batches = []
        
    def on_batch_end(self, batch, logs={}):
        current_batch = logs.get('batch')
        self.batches.append(current_batch)
        #print(current_batch)
        if current_batch >= self.max_batches or self.lr >= self.max_lr:
            self.model.stop_training = True
        else:
            self.lr = self.lr + (current_batch * self.lr_step_size)
            K.set_value(self.model.optimizer.lr, self.lr)
            self.losses.append(logs.get('loss'))
            self.lrs.append(self.lr)
    
    def on_train_end(self, logs = None):
        plt.rcParams["figure.figsize"] = (20,10)
        plt.plot(self.lrs, self.losses)
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.title('learning rate finder curve')
        plt.show()

history = LossHistory()
lrf = LRFinder()
model.fit_generator(train_generator, epochs =2, steps_per_epoch = 316 ,
                    callbacks = [lrf], workers = 4)

sgd = keras.optimizers.SGD(lr = 0.002, momentum = 0.9)
adam = keras.optimizers.Adam(lr = 0.001)
model.compile(optimizer = adam , loss = 'mse' , metrics = [mae])
model.fit_generator(train_generator, epochs =5, validation_data = (test_x,test_y), steps_per_epoch = 158 ,
                    callbacks = [checkpoint], workers = 4)
model.load_weights('best_weights.hd5')
model.fit_generator(train_generator, epochs =5, validation_data = (test_x,test_y), steps_per_epoch = 158,
                    callbacks = [checkpoint], workers = 4)
#preds = model.predict(next(valid_generator))
#imgs,labels = next(fixed_val)
preds = model.predict(test_x) * boneage_std + boneage_mean
labels = test_y*boneage_std + boneage_mean
print(preds.shape)
fig, ax1 = plt.subplots(1,1, figsize = (6,6))
ax1.plot(labels, preds, 'r.', label = 'predictions')
ax1.plot(labels, labels, 'b-', label = 'actual')
ax1.legend()
ax1.set_xlabel('Actual Age (Months)')
ax1.set_ylabel('Predicted Age (Months)')
preds.max(),preds.min()
