from os import listdir
from os.path import isfile, join
from shutil import copy
import random, os
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from shutil import rmtree
#rmtree('temp')
#rmtree('voice_dataset')
SOURCE    = '../input/speech-in-russian-digits-from-0-to-9/voice_0-9_RU/'
DIR       = 'voice_dataset'
TRAIN_DIR = '/train/'
VALID_DIR = '/valid/'
TEMP      = 'temp/'

#for dirct in os.listdir(os.path.join('temp')):
#rmtree('temp')
#rmtree('voice_dataset')
#print(len(SOURCE))
def convert_to_spectrogram(sours_dir, to_safe_dir, spectrogram_dimensions=(34, 50)):
    file_names = [f for f in listdir(sours_dir) if isfile(join(sours_dir, f)) and '.wav' in f]
    for file_name in file_names:
        try:
            sample_rate, samples = wav.read(sours_dir + file_name)     
            fig = plt.figure()
            fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.specgram(samples, cmap='gray_r', Fs=2, noverlap=16)
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            fig.savefig(to_safe_dir + file_name.replace('.wav', '.png'), bbox_inches="tight", pad_inches=0)
            plt.close()
        except UnboundLocalError:
            print("Problem with " , sours_dir ,  " file")
            continue  
if not os.path.exists(TEMP):
    os.makedirs(TEMP)
    convert_to_spectrogram( SOURCE, TEMP ) 
try:
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        os.makedirs(DIR+TRAIN_DIR)
        os.makedirs(DIR+VALID_DIR)
        for digit in range(0,10):
            os.makedirs(DIR+TRAIN_DIR+str(digit))
            os.makedirs(DIR+VALID_DIR+str(digit))
    print("Folder " , DIR ,  " is created") 
except FileExistsError:
    print("Problem with " , DIR ,  " folder")
if os.path.exists(TEMP):
    voice_dataset = os.listdir(TEMP)
    random.shuffle(voice_dataset)
else:
    print("Problem with " , TEMP ,  " folder")
dir_dig = {digit:list() for digit in range(0,10)}
for item in voice_dataset:
    for dig in range(0, 10):
        if dig == int(item[0]):
            dir_dig[dig].append(item) 
split = 0.75

for dig in range(0, 10):
    digit_train, digit_test = dir_dig[dig][:int( split*len(dir_dig[dig]) )], dir_dig[dig][int( split*len(dir_dig[dig]) ):]
    for train, test in zip(digit_train, digit_test):
        copy(TEMP+str(train), DIR+TRAIN_DIR+str(dig))
        copy(TEMP+str(test), DIR+VALID_DIR+str(dig)) 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import optimizers 
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
#Define Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(34, 50, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
#Compile

model.summary()

model.compile(
    optimizer=optimizers.rmsprop(),
    loss='categorical_crossentropy',
    metrics = ['acc']
)
train_data = ImageDataGenerator( 1./255 )
test_data  = ImageDataGenerator( 1./255 )

filepath='cnn_spectrogram_to_digit.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.1, min_lr=0.0001)

callbacks_list=[checkpoint, learning_rate_reduction]

train_gen = train_data.flow_from_directory(
    DIR+TRAIN_DIR,
    target_size = (34, 50),
    batch_size = 20,
    class_mode = 'categorical' 
)
test_gen = test_data.flow_from_directory(
    DIR+VALID_DIR,
    target_size = (34, 50),
    batch_size = 20,
    class_mode = 'categorical' 
)
history = model.fit_generator(
    train_gen,
    steps_per_epoch=100,
    epochs=1000,
    callbacks=callbacks_list,
    validation_data=test_gen,
    validation_steps=100
)



test_loss, test_acc = model.evaluate_generator(test_gen, steps=100)

#model.save('cnn_spectrogram_to_digit.h5')

acc = history.history['acc']
val_acc =  history.history['val_acc']


loss = history.history['loss']
val_loss =  history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Valodation acc')
plt.title('Training and validation acc')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Valodation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
print('Test acc:', test_acc)