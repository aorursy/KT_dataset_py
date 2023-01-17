import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
%matplotlib inline 

from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'

ROWS = 224
COLS = 224
CHANNELS = 3
print(os.listdir(TRAIN_DIR)[:5])
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
train_images = train_dogs[:] + train_cats[:]
random.shuffle(train_images)
test_images =  test_images[:]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))
labels = []
for i in train_images:
    if 'dog' in i.split('/')[-1]:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels).set_title('Cats and Dogs')


def show_cats_and_dogs(idx):
    cat = read_image(train_cats[idx])
    dog = read_image(train_dogs[idx])
    pair = np.concatenate((cat, dog), axis=1)
    print(cat.shape)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
    
for idx in range(0,5):
    show_cats_and_dogs(idx)
dog_avg = np.array([dog[0].T for i, dog in enumerate(train) if labels[i]==1]).mean(axis=0)
plt.imshow(dog_avg)
plt.title('Your Average Dog')


cat_avg = np.array([cat[0].T for i, cat in enumerate(train) if labels[i]==0]).mean(axis=0)
plt.imshow(cat_avg)
plt.title('Your Average Cat')


from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization

optimizer = RMSprop(lr=1e-4)
objective = 'binary_crossentropy'


def catdog():
    
    model = Sequential()

    model.add(Convolution2D(96, (11, 11),strides=(4,4), padding='same', input_shape=(3, ROWS, COLS), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format = "channels_first"))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((2,2)))
    
    model.add(Convolution2D(128, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format = "channels_first"))
    model.add(BatchNormalization())
    model.add(ZeroPadding2D((1,1)))
    
    model.add(Convolution2D(384, (3, 3), padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    
    model.add(Convolution2D(192, (3, 3), padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    
    model.add(Convolution2D(192, (3, 3), padding='same', activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2, data_format = 'channels_first'))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=objective, optimizer='sgd', metrics=['accuracy'])
    return model
    
model = catdog()

nb_epoch = 10
batch_size = 16

## Callback for loss logging per epoch
class LossHistory(Callback):
    
    
    def on_train_begin(self, logs={}):
        self.x = 0
        self.losses = []
        self.val_losses = []
        print('Epoch ' + str(self.x) + '/' + str(int(len(train)/nb_epoch)))
        print(logs)
        
    def on_epoch_end(self, batch, logs={}):
        self.x+=1
        print('Epoch ' + str(self.x) + '/' + str(int(len(train)/batch_size)))
        print(logs)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        model.save('my_model.h5')
        
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')        
        
def run_catdog():
    
    history = LossHistory()
    model.fit(train, labels, batch_size=batch_size, epochs=nb_epoch,
              validation_split=0.2, verbose=0, shuffle=True, callbacks=[history, early_stopping])
    

    predictions = model.predict(test, verbose=0)
    return predictions, history

predictions, history = run_catdog()
loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()

# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on

from keras.models import load_model
from keras.preprocessing import image
import pandas as pd
import numpy as np

# dimensions of our images
#img_width, img_height = 224, 224

# load the model we saved

#model.load_weights('../input/model/my_model.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# predicting images
ids = []
labels = []
df = []
#for i in range(len(test_images)):
for i in range(len(test_images)):    
    #fig = read_image(test_images[i])
    img = test[i]
    #img = image.load_img('test', target_size=(img_width, img_height))
    x = np.expand_dims(img, axis=0)
    '''
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    '''
    
    
    # pass the list of multiple images np.vstack()
    e = {'id':test_images[i].split('/')[-1].split('.')[0],
    'label':predictions[i][0]}
    if i%250==0:
        print('{} out of {} images labeled.'.format(i, len(test_images)))
    df.append(e)
    # print the classes, the images belong to
    '''
    print (classes)
    plt.figure(figsize=(10,5))
    plt.imshow(fig)
    plt.show()
    '''
    
df = pd.DataFrame(df)
print(df.head())

df = df.apply(pd.to_numeric, errors='ignore').sort_values('id')

df.index = np.arange(1, len(df)+1)
df = df.drop(columns = 'id')
df.index.name = 'id'
df.to_csv('submission.csv')


df.head(10)
for i in range(5,20):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
        
    plt.imshow(test[i].T)
    plt.show()
print(predictions[:20])