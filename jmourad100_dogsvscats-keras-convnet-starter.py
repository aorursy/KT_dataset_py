import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Model
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation 
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
import os, cv2, random

# %matplotlib inline
# import keras
# keras.backend.backend() #=> Tensorflow
# This function resizes the images to 64x64 and samples 2000 images (8%) of the data.
# I also separated cats and dogs for exploratory analysis

TRAIN_DIR = '../input/train'
TEST_DIR = '../input/test'
ROWS = 64
COLS = 64
CHANNELS = 3

train_images = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR)]
train_dogs = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats = [TRAIN_DIR+"/"+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images = [TEST_DIR+"/"+i for i in os.listdir(TEST_DIR)]

# train_images = train_dogs[:1000] + train_cats[:1000]
# random.shuffle(train_images)
# test_images = test_images[:25]

def read_image(file_path):
    img= cv2.imread(file_path, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS, CHANNELS), dtype=np.uint8)
    
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
        if i%1500 == 0: print('Processed {} of {}'.format(i, count))
    return data
train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))
# We're dealing with classification problem here - (1) dogs (0) cats
labels = []
for i in train_images:
    if 'dog' in i:
        labels.append(1)
    else:
        labels.append(0)
        
sns.countplot(labels)
plt.title('Cats and Dogs');
# A quick side-by-side comparison of the animals
def show_cats_and_dogs(idx):
    cat = read_image(train_images[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(15, 5))
    plt.imshow(pair)
    plt.show()

for idx in range(2):
    show_cats_and_dogs(idx)
def build_model(N_Filters=32):
    input_layer = Input((ROWS, COLS, CHANNELS), name="InputLayer")
    # Block 1
    x = Convolution2D(N_Filters*1, (3,3), padding='same', activation='relu', name='block1_conv1')(input_layer)
    x = Convolution2D(N_Filters*1, (3,3), padding='same', activation='relu', name='block1_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block1_pool')(x)
    
    # Block 2
    x = Convolution2D(N_Filters*2, (3,3), padding='same', activation='relu', name='block2_conv1')(x)
    x = Convolution2D(N_Filters*2, (3,3), padding='same', activation='relu', name='block2_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block2_pool')(x)
    
    # Block 3
    x = Convolution2D(N_Filters*4, (3,3), padding='same', activation='relu', name='block3_conv1')(x)
    x = Convolution2D(N_Filters*4, (3,3), padding='same', activation='relu', name='block3_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block3_pool')(x)
    
    # Block 4
    x = Convolution2D(N_Filters*8, (3,3), padding='same', activation='relu', name='block4_conv1')(x)
    x = Convolution2D(N_Filters*8, (3,3), padding='same', activation='relu', name='block4_conv2')(x)
    x = MaxPooling2D((2,2), strides=(2,2), name='block4_pool')(x)
    
    x = Flatten(name='flatten')(x)
    x = Dense(N_Filters*8, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(N_Filters*8, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(input_layer, output)
    model.compile(optimizer=RMSprop(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model
build_model().summary()
## Callback for loss loggingper epoch
batch_size=16; epochs=10
model = build_model()
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
    
early_stopping = EarlyStopping(monitor = 'val_loss', patience=3, verbose=1,mode='auto')
def run_catdog(batch_size=16, epochs=20):
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, epochs=epochs,
              validation_split=0.25, verbose=1, shuffle=True, callbacks=[history, early_stopping])
    predictions = model.predict(test, verbose=1)
    return predictions, history
predictions, history = run_catdog()
loss = history.losses
val_loss = history.val_losses

plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.title('VGG-16 Loss Trend')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0,epochs)[0::2])
plt.legend();
for i in range(4):
    if predictions[i][0] >= 0.5:
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else:
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
    plt.imshow(test[i])
    plt.show()
model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')
