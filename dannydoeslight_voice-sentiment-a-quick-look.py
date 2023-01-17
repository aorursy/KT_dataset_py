# Import libraries 

import os

import sys

import warnings



import numpy as np

import pandas as pd



import librosa



import matplotlib.pyplot as plt

import seaborn as sns



# Keras

import keras

from keras.models import Sequential

from keras.layers import Dense, Input, Flatten, Dropout, Activation, BatchNormalization

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.utils import np_utils, to_categorical



# sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



# ignore warnings 

if not sys.warnoptions:

    warnings.simplefilter("ignore")

warnings.filterwarnings("ignore", category=DeprecationWarning) 
#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"

RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"

SAVEE = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"

CREMA = "/kaggle/input/cremad/AudioWAV/"



# Run one example 

dir_list = os.listdir(SAVEE)

dir_list[0:9]
# Get the data location for SAVEE

dir_list = os.listdir(SAVEE)



# parse the filename to get the emotions

emotion=[]

path = []

for i in dir_list:

    if i[-8:-6]=='_a':

        emotion.append('male_angry')

    elif i[-8:-6]=='_d':

        emotion.append('male_disgust')

    elif i[-8:-6]=='_f':

        emotion.append('male_fear')

    elif i[-8:-6]=='_h':

        emotion.append('male_happy')

    elif i[-8:-6]=='_n':

        emotion.append('male_neutral')

    elif i[-8:-6]=='sa':

        emotion.append('male_sad')

    elif i[-8:-6]=='su':

        emotion.append('male_surprise')

    else:

        emotion.append('male_error') 

    path.append(SAVEE + i)

    

# Now check out the label count distribution 

SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])

SAVEE_df['source'] = 'SAVEE'

SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)

SAVEE_df.labels.value_counts()
dir_list = os.listdir(RAV)

dir_list.sort()



emotion = []

gender = []

path = []

for i in dir_list:

    fname = os.listdir(RAV + i)

    for f in fname:

        part = f.split('.')[0].split('-')

        emotion.append(int(part[2]))

        temp = int(part[6])

        if temp%2 == 0:

            temp = "female"

        else:

            temp = "male"

        gender.append(temp)

        path.append(RAV + i + '/' + f)



        

RAV_df = pd.DataFrame(emotion)

RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})

RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)

RAV_df.columns = ['gender','emotion']

RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion

RAV_df['source'] = 'RAVDESS'  

RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)

RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)

RAV_df.labels.value_counts()
dir_list = os.listdir(TESS)

dir_list.sort()

dir_list
path = []

emotion = []



for i in dir_list:

    fname = os.listdir(TESS + i)

    for f in fname:

        if i == 'OAF_angry' or i == 'YAF_angry':

            emotion.append('female_angry')

        elif i == 'OAF_disgust' or i == 'YAF_disgust':

            emotion.append('female_disgust')

        elif i == 'OAF_Fear' or i == 'YAF_fear':

            emotion.append('female_fear')

        elif i == 'OAF_happy' or i == 'YAF_happy':

            emotion.append('female_happy')

        elif i == 'OAF_neutral' or i == 'YAF_neutral':

            emotion.append('female_neutral')                                

        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':

            emotion.append('female_surprise')               

        elif i == 'OAF_Sad' or i == 'YAF_sad':

            emotion.append('female_sad')

        else:

            emotion.append('Unknown')

        path.append(TESS + i + "/" + f)



TESS_df = pd.DataFrame(emotion, columns = ['labels'])

TESS_df['source'] = 'TESS'

TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)

TESS_df.labels.value_counts()
dir_list = os.listdir(CREMA)

dir_list.sort()

print(dir_list[0:10])
gender = []

emotion = []

path = []

female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,1029,1030,1037,1043,1046,1047,1049,

          1052,1053,1054,1055,1056,1058,1060,1061,1063,1072,1073,1074,1075,1076,1078,1079,1082,1084,1089,1091]



for i in dir_list: 

    part = i.split('_')

    if int(part[0]) in female:

        temp = 'female'

    else:

        temp = 'male'

    gender.append(temp)

    if part[2] == 'SAD' and temp == 'male':

        emotion.append('male_sad')

    elif part[2] == 'ANG' and temp == 'male':

        emotion.append('male_angry')

    elif part[2] == 'DIS' and temp == 'male':

        emotion.append('male_disgust')

    elif part[2] == 'FEA' and temp == 'male':

        emotion.append('male_fear')

    elif part[2] == 'HAP' and temp == 'male':

        emotion.append('male_happy')

    elif part[2] == 'NEU' and temp == 'male':

        emotion.append('male_neutral')

    elif part[2] == 'SAD' and temp == 'female':

        emotion.append('female_sad')

    elif part[2] == 'ANG' and temp == 'female':

        emotion.append('female_angry')

    elif part[2] == 'DIS' and temp == 'female':

        emotion.append('female_disgust')

    elif part[2] == 'FEA' and temp == 'female':

        emotion.append('female_fear')

    elif part[2] == 'HAP' and temp == 'female':

        emotion.append('female_happy')

    elif part[2] == 'NEU' and temp == 'female':

        emotion.append('female_neutral')

    else:

        emotion.append('Unknown')

    path.append(CREMA + i)

    

CREMA_df = pd.DataFrame(emotion, columns = ['labels'])

CREMA_df['source'] = 'CREMA'

CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)

CREMA_df.labels.value_counts()
df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)

print(df.labels.value_counts())

df.to_csv("Data_path.csv",index=False)
data_path = pd.read_csv("/kaggle/working/Data_path.csv")

data_path.head()
plt.figure(figsize=(20, 8))

sns.countplot('labels', data=df)
# This takes a few minutes (~15 mins): 

df = pd.DataFrame(columns=['feature'])



# feature extraction 

counter=0

for index,path in enumerate(data_path.path):

    X, sample_rate = librosa.load(path

                                  , res_type='kaiser_fast'

                                  ,duration=2.5

                                  ,sr=44100

                                  ,offset=0.5

                                 )

    sample_rate = np.array(sample_rate)

    

    # mean as the feature. Could do min and max etc as well. 

    mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                        sr=sample_rate, 

                                        n_mfcc=13),

                    axis=0)

    df.loc[counter] = [mfccs]

    counter=counter+1   



# Check a few records to make sure its processed successfully

print(len(df))

df.head()
df = pd.concat([data_path, pd.DataFrame(df['feature'].values.tolist())],axis=1)

df[:5]
df=df.fillna(0)

print(df.shape)

df[:5]
df.to_csv("Data_mfcc.csv",index=False)
df = pd.read_csv("/kaggle/working/Data_mfcc.csv")

df.head() 
facet = sns.FacetGrid(df, hue="labels", aspect=4)

facet.map(sns.kdeplot,'207', shade= True)

facet.set(xlim=(0, df['207'].max()))

facet.add_legend()
facet = sns.FacetGrid(df, hue="labels", aspect=4)

facet.map(sns.kdeplot,'208', shade= True)

facet.set(xlim=(0, df['208'].max()))

facet.add_legend()
facet = sns.FacetGrid(df, hue="labels", aspect=4)

facet.map(sns.kdeplot,'111', shade= True)

facet.set(xlim=(0, df['111'].max()))

facet.add_legend()
facet = sns.FacetGrid(df, hue="labels", aspect=4)

facet.map(sns.kdeplot,'150', shade= True)

facet.set(xlim=(0, df['150'].max()))

facet.add_legend()
# Split between train and test 

y = df.labels

X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','labels','source'],axis=1)

                                                    , y

                                                    , test_size=0.25

                                                    , stratify = y #shuffle=True

                                                    , random_state=42

                                                   )



# Quick Check: before

X_train[110:130]
# Data normalization 

mean = np.mean(X_train, axis=0)

std = np.std(X_train, axis=0)



X_train = (X_train - mean)/std

X_test = (X_test - mean)/std



# Quick Check: after

X_train[110:130]
# prepare the format for Keras 

X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)



# one hot encode the target 

lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))



print(X_train.shape)

print(lb.classes_)
# Using a CNN, we need to specify the 3rd dimension: 1 because we're doing a 1D CNN. 

X_train = np.expand_dims(X_train, axis=2)

X_test = np.expand_dims(X_test, axis=2)

X_train.shape
# building the model:

model = Sequential()

model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns

model.add(Activation('relu'))

model.add(Conv1D(256, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(128, 8, padding='same'))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Conv1D(64, 8, padding='same'))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(14)) # Target class number

model.add(Activation('softmax'))



model.summary()
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



# feel free to try the optimizers bellow for better results:

# opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

# opt = keras.optimizers.Adam(lr=0.0001)

# opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)

# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience=2, verbose=1)
model_history=model.fit(X_train, y_train, batch_size=16, epochs=15, validation_data=(X_test, y_test), callbacks=[early_stopping])
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test accuracy {test_acc * 100:.2f} %")

print(f"Test loss {test_loss}")
pd.DataFrame(model_history.history).plot()