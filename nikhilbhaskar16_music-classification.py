# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore", category = FutureWarning)



import IPython.display as ipd

import numpy as np

import pandas as pd

import librosa

import matplotlib.pyplot as plt

import os

import librosa.display

from scipy.io import wavfile as wav

from sklearn import metrics 

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import Adam

from keras.utils import to_categorical
music_list = filenames
hop_length = 1024
sr = 22050
np_music=[]

i=1

for each in music_list:

    if i==10:

        break

    plt.figure(figsize=(8, 4))

    files = '../input/'+each

    audio, sample_rate = librosa.load(files)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=hop_length) # hop_length

    R = librosa.segment.recurrence_matrix(mfcc,sym = True)

    R_aff = librosa.segment.recurrence_matrix(mfcc, mode='affinity')

        

    plt.subplot(1, 2, 1)

    librosa.display.specshow(R, x_axis='time', y_axis='time', hop_length=hop_length)

    plt.title(f'Binary recurrence (symmetric)')

    plt.subplot(1, 2, 2)

#     plt.show()

    librosa.display.specshow(R_aff, x_axis='time', y_axis='time',hop_length=hop_length, cmap='magma_r')

    plt.title(f'Affinity recurrence')

    plt.tight_layout()

    plt.show()

    i+=1

#     np_music.append(audio)
def extract_features(file_name):

    file_path = "../input/" + file_name

    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast') 

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    mfccs_processed = np.mean(mfccs.T,axis=0)

     

    return mfccs_processed
label_list=[]

i=0

# file = h5py.File('music13.hdf5','w') #change name of file everytime before run

for each in music_list:

#     print(each)

    if i<10:

#         dset = file.create_dataset("Alpha_"+str(i),data=each)

        label_list.append("Alpha")

    if i>=10 and i<22:

#         dset = file.create_dataset("Beta_"+str(i),data = each)

        label_list.append("Beta")

    if i>=22:

#         dset = file.create_dataset("Delta_"+str(i),data = each)

        label_list.append("Delta")

#     print('here')    

    i+=1

label_list  
features = []

# Iterate through each sound file and extract the features 

for each in music_list:

    

#     class_label = row["class"]

    data = extract_features(each)

    

    features.append(data)
# Convert into a Panda dataframe 

featuresdf = pd.DataFrame({'feature':features,'class_label':label_list})

featuresdf.head()
# Convert features and corresponding classification labels into numpy arrays

X = np.array(featuresdf.feature.tolist())

y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels

le = LabelEncoder()

yy = to_categorical(le.fit_transform(y))
# split the dataset 

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.1, random_state = 127)
num_labels = yy.shape[1]

# filter_size = def build_model_graph(input_shape=(40,)):

model = Sequential()

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(256))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_labels))

model.add(Activation('softmax'))

    # Compile the model

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

#     return modelmodel = build_model_graph()
num_epochs = 100

num_batch_size = 32

# model = Sequential()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)
# Evaluating the model on the training and testing set

score = model.evaluate(x_train, y_train, verbose=0)

print("Training Accuracy: {0:.2%}".format(score[1]))

score = model.evaluate(x_test, y_test, verbose=0)

print("Testing Accuracy: {0:.2%}".format(score[1]))