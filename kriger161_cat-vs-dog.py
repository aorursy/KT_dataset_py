import numpy as np
import matplotlib.pyplot as plt
import os, cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm      # a nice pretty percentage bar for tasks.
from random import shuffle
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 128
COLS = 128
CHANNELS = 1
RETRAIN = True

data_dict = {}

dog_train_list = [TRAIN_DIR+name for i,name in \
                  enumerate(os.listdir(TRAIN_DIR))
                  if 'dog' in name]

cat_train_list = [TRAIN_DIR+name for i,name in \
                  enumerate(os.listdir(TRAIN_DIR))
                  if 'cat' in name]

TRAIN_COUNT = len(dog_train_list) + len(cat_train_list)


data_dict['train_data_files'] = dog_train_list + cat_train_list
data_dict['train_labels'] = [1]*int(TRAIN_COUNT/2) + [0]*int(TRAIN_COUNT/2)

# Loading the test set
test_image_list = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


print('Total Training Images: {}'.format(len(data_dict['train_data_files'])))
print('Total Test Images: {}'.format(len(test_image_list)))

def load_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (ROWS,COLS))
    

def load_image_list(file_list):
    count = len(file_list)
    data = np.ndarray((count, ROWS, COLS, CHANNELS),\
                      dtype=np.uint8)
    for i, image_name in tqdm(enumerate(file_list)):
        data[i] = np.expand_dims(load_image(image_name), axis=2)
    return data

data_dict['train_data'] = load_image_list(\
                            data_dict['train_data_files'])

test_set = load_image_list(test_image_list)
print('data_dict shape: {}'.format(data_dict['train_data'].shape))
print('test_set shape: {}'.format(test_set.shape))
datagen_train = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
     zoom_range = 0.1,
    )

datagen_val = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    )
f, axarr = plt.subplots(2,2)

axarr[0,0].imshow(np.squeeze(data_dict['train_data'][0], axis=2), cmap='gray')

axarr[1,0].imshow(np.squeeze(data_dict['train_data'][int(TRAIN_COUNT/2)-1], axis=2), cmap='gray')

axarr[0,1].imshow(np.squeeze(data_dict['train_data'][int(TRAIN_COUNT/2) + 2], axis=2), cmap='gray')

axarr[1,1].imshow(np.squeeze(data_dict['train_data'][TRAIN_COUNT -1], axis=2), cmap='gray')
plt.show()
model = Sequential()
model.add(Conv2D(filters=4, kernel_size=(4,4),\
                 padding='Same', activation='relu',\
                 input_shape=(ROWS,COLS,CHANNELS)))
model.add(Conv2D(filters=8, kernel_size=(4,4),\
                 padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(4,4)))
# model.add(Dropout(0.5))

model.add(Conv2D(filters=16, kernel_size=(8,8),\
                 padding='Same', activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(8,8),\
                 padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(4,4)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(8**2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

optimizer = SGD(lr=0.03)
model.compile(optimizer=optimizer,
              loss="binary_crossentropy",
              metrics=["accuracy"])

print(model.summary())
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Saving the initial weights, in order to initiate the model in each fold.
model.save_weights('initial.h5')

filepath='cat_dog_v4a1'

early_stop = EarlyStopping(monitor='val_loss',
                              patience=5,
                              verbose=0, 
                              mode='min')

checkpoint = ModelCheckpoint(filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=False,
                             mode='min')

epochs = 60

kfold = StratifiedKFold(n_splits = 4, shuffle=True)
history_list = []
for i, (train, test) in enumerate(kfold.split(data_dict['train_data'], data_dict['train_labels'])):
    model.load_weights('initial.h5')
    # Training 
    datagen_train.fit(data_dict['train_data'][train], augment=True)
    datagen_val.fit(data_dict['train_data'][train], augment=False)
    
    history = model.fit_generator(generator=datagen_train.flow(
                      data_dict['train_data'][train],np.asarray(data_dict['train_labels'])[train]),
                  epochs=epochs,
                  verbose=1,
                  validation_data=datagen_val.flow(data_dict['train_data'][test]\
                                                   ,np.asarray(data_dict['train_labels'])[test]),
                  callbacks=[checkpoint,early_stop])
    
    history_list.append(history)
    
    # Evaluate the test set with the current fold model
    filename = 'sub_file' + str(i) +'.csv'
    with open(filename,'w') as f:
        f.write('id,label\n')
        print('Evaluate the test set #{}..'.format(i))
        prediction = model.predict_generator(datagen_val.flow(test_set), steps=391)
        print('Writing the prediction in the submition file..')
        for i, image_file in tqdm(enumerate(prediction)):
            f.write('{},{}\n'.format(i+1, prediction[i][0]))
                           

print(len(history_list))
train_loss = np.array([history.history['loss'] for history in history_list])
train_loss_mean = np.mean(train_loss,axis=0)

val_loss = np.array([history.history['val_loss'] for history in history_list])
val_loss_mean = np.mean(val_loss,axis=0)

train_acc = np.array([history.history['acc'] for history in history_list])
train_acc_mean = np.mean(train_acc,axis=0)

val_acc = np.array([history.history['val_acc'] for history in history_list])
val_acc_mean = np.mean(val_acc,axis=0)

fig, ax = plt.subplots(2,2)
for i, history in enumerate(history_list):
    ax[i%2, int(i/2)].plot(history.history['loss'], color='b', label="Training Loss")
    ax[i%2, int(i/2)].plot(history.history['val_loss'], color='r', label='Validation Loss')
fig, ax = plt.subplots(2,1)
# Loss Plot
# for history in history_list
ax[0].plot(train_loss_mean, color='b', label="Training Loss")
ax[0].plot(val_loss_mean, color='r', label='Validation Loss')

legend = ax[0].legend(loc='best', shadow=True)

# Accuracy Plot
ax[1].plot(train_acc_mean, color='b', label='Training Accuracy')
ax[1].plot(val_acc_mean, color='r', label='Validation Accuracy')
legend = ax[1].legend(loc='best', shadow=True)
# test_image_list = [TEST_DIR+i for i in os.listdir(TEST_DIR)]

# datagen_test = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True)

# test_set = load_image_list(test_image_list)
# datagen_test.fit(test_set)
# with open('submission_file2.csv','w') as f:
#     f.write('id,label\n')
# with open('submission_file2.csv','a') as f:
# #     test_set = load_image_list(test_image_list)
#     print('Evaluate the test set..')
#     prediction = model.predict_generator(datagen_test.flow(test_set), steps=391)
#     print('Writing the prediction in the submition file..')
#     for i, image_file in tqdm(enumerate(prediction)):
#         f.write('{},{}\n'.format(i+1, prediction[i][0]))
