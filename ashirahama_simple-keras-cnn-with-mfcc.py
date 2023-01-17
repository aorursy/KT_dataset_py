import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc



%matplotlib inline

import matplotlib.pyplot as plt

import librosa.display

import librosa

import IPython.display as ipd

print(os.listdir("../input"))
gc.collect()
# print(os.listdir("../input/train_curated"))
TRAIN_NOISY_PATH = "../input/train_noisy.csv"

TRAIN_CURATED_PATH = "../input/train_curated.csv"

TRAIN_NOISY = "../input/train_noisy/"

TRAIN_CURATED = "../input/train_curated/"

TEST = "../input/test/"

SUB_PATH = "../input/sample_submission.csv"



train_noisy = pd.read_csv(TRAIN_NOISY_PATH)

train_curated = pd.read_csv(TRAIN_CURATED_PATH)

sub = pd.read_csv(SUB_PATH)



SAMPLING_RATE = 44100

MFCC_NUM = 20

MFCC_MAX_LEN = 2000
target_labels = ['Accelerating_and_revving_and_vroom','Accordion','Acoustic_guitar','Applause','Bark','Bass_drum','Bass_guitar','Bathtub_(filling_or_washing)','Bicycle_bell','Burping_and_eructation','Bus','Buzz','Car_passing_by','Cheering','Chewing_and_mastication','Child_speech_and_kid_speaking','Chink_and_clink','Chirp_and_tweet','Church_bell','Clapping','Computer_keyboard','Crackle','Cricket','Crowd','Cupboard_open_or_close','Cutlery_and_silverware','Dishes_and_pots_and_pans','Drawer_open_or_close','Drip','Electric_guitar','Fart','Female_singing','Female_speech_and_woman_speaking','Fill_(with_liquid)','Finger_snapping','Frying_(food)','Gasp','Glockenspiel','Gong','Gurgling','Harmonica','Hi-hat','Hiss','Keys_jangling','Knock','Male_singing','Male_speech_and_man_speaking','Marimba_and_xylophone','Mechanical_fan','Meow','Microwave_oven','Motorcycle','Printer','Purr','Race_car_and_auto_racing','Raindrop','Run','Scissors','Screaming','Shatter','Sigh','Sink_(filling_or_washing)','Skateboard','Slam','Sneeze','Squeak','Stream','Strum','Tap','Tick-tock','Toilet_flush','Traffic_noise_and_roadway_noise','Trickle_and_dribble','Walk_and_footsteps','Water_tap_and_faucet','Waves_and_surf','Whispering','Writing','Yell','Zipper_(clothing)']
def count_labels(labels):

    array_lbs = labels.split(",")

    return len(array_lbs)



def count_target_labels(labels):

    count = 0

    array_lbs = labels.split(",")

    for lb in array_lbs:

        if lb in target_labels:

            count += 1

    return count
train_noisy["label_count"] = train_noisy["labels"].apply(count_labels)

train_noisy["target_label_count"] = train_noisy["labels"].apply(count_target_labels)
train_noisy.head(10)
print("Count train_noisy:" + str(train_noisy.shape[0]))

print("Count records without target label in train_noisy:" + str(train_noisy.query("target_label_count == 0").shape[0]))
ipd.Audio(TRAIN_NOISY + train_noisy["fname"][8])
train_curated["label_count"] = train_curated["labels"].apply(count_labels)

train_curated["target_label_count"] = train_curated["labels"].apply(count_target_labels)
train_curated.head(10)
print("Count train_curated:" + str(train_curated.shape[0]))

print("Count records without target label in train_curated:" + str(train_curated.query("target_label_count == 0").shape[0]))
ipd.Audio(TRAIN_CURATED + train_curated["fname"][8])
sub.head()
print("Count test:" + str(sub.shape[0]))
ipd.Audio(TEST + sub["fname"][4])
train_curated.groupby("labels").size()
import librosa

import os

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import numpy as np

from tqdm import tqdm
# test, sr = librosa.load(TRAIN_CURATED + train_curated["fname"][3], sr=SAMPLING_RATE)

# librosa.feature.mfcc(test, n_mfcc=128, sr=44100).shape
# def wav2mfcc(file_path, max_len=11):

def wav2mfcc(wave, max_len=MFCC_MAX_LEN):

#     mfcc = librosa.feature.mfcc(wave, sr=16000)

    mfcc = librosa.feature.mfcc(wave, n_mfcc=MFCC_NUM, sr=SAMPLING_RATE)



    # If maximum length exceeds mfcc lengths then pad the remaining ones

    if (max_len > mfcc.shape[1]):

        pad_width = max_len - mfcc.shape[1]

        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')



    # Else cutoff the remaining parts

    else:

        mfcc = mfcc[:, :max_len]

    

    return mfcc
def get_label_num(labels):

    lbs = labels.split(",")

#     target_lb = "Accelerating_and_revving_and_vroom"

    target_arr = np.zeros(80)

    for lb in lbs:

        if(lb in target_labels):

            i = target_labels.index(lb)

            target_arr[i] = 1

            break

    return target_arr
X = []

y = []



def append_X_Y(labels, wave):

    y.append(get_label_num(labels))

    mfcc = wav2mfcc(wave)

    X.append(mfcc)



for index, row in tqdm(train_curated.iterrows()):

    labels = row["labels"]

    wave, sr = librosa.load(TRAIN_CURATED + row["fname"], mono=True, sr=44100)

    wave = wave[::3]

    

#     if(len(labels.split(",")) == 1):

    append_X_Y(labels, wave)

        

# for index, row in tqdm(train_noisy.iterrows()):

#     labels = row["labels"]

#     wave, sr = librosa.load(TRAIN_NOISY + row["fname"], mono=True, sr=None)

#     wave = wave[::3]

#     append_X_Y(labels, wave)



# np.save('train_augumented_mfcc_vectors.npy', X)

# np.save('train_augumented_labels.npy', y)
gc.collect()
X = np.array(X)

y = np.array(y)

X.shape[0] == len(y)
# y_hot = to_categorical(y)

y_hot = y
X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size= 0.2, random_state=True, shuffle=True)
X_train.shape
# Feature dimension

feature_dim_1 = MFCC_NUM

# Second dimension of the feature is dim2

feature_dim_2 = MFCC_MAX_LEN

channel = 1

epochs = 70

batch_size = 100

verbose = 1

num_classes = len(target_labels)

# Reshaping to perform 2D convolution

X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)

X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)



y_train_hot = y_train

y_test_hot = y_test
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.utils import to_categorical

from keras import optimizers
def get_model():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))

    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))

    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.25))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax'))

    return model
model = get_model()



optimizer = optimizers.SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)

# optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=optimizer,

              metrics=['accuracy'])

model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
sub = pd.read_csv("../input/sample_submission.csv")



for index, row in tqdm(sub.iterrows()):

    wave, sr = librosa.load(TEST + row["fname"], mono=True, sr=None)

    wave = wave[::2]

    

    mfcc = wav2mfcc(wave)

    X_test = mfcc.reshape(1, feature_dim_1, feature_dim_2, channel)

    preds = model.predict(X_test)[0]

    

    for i, col in enumerate(target_labels):

        sub.loc[index, col] = preds[i]
sub.to_csv("submission.csv",index=False)