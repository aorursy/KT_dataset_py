# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import librosa

import librosa.display

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import keras

from keras.preprocessing import sequence

from keras.models import Sequential

from keras.layers import Dense, Embedding

from keras.layers import LSTM

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.layers import Input, Flatten, Dropout, Activation

from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from keras.models import Model

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import confusion_matrix
from keras import regularizers

import os

mylist= os.listdir('../input/ravdess-song-speech/RawData')

print(mylist)
feeling_list=[]

for item in mylist:

    if item[6:-16]=='02' and int(item[18:-4])%2==0:

        feeling_list.append('female_calm')

    elif item[6:-16]=='02' and int(item[18:-4])%2==1:

        feeling_list.append('male_calm')

    elif item[6:-16]=='03' and int(item[18:-4])%2==0:

        feeling_list.append('female_happy')

    elif item[6:-16]=='03' and int(item[18:-4])%2==1:

        feeling_list.append('male_happy')

    elif item[6:-16]=='04' and int(item[18:-4])%2==0:

        feeling_list.append('female_sad')

    elif item[6:-16]=='04' and int(item[18:-4])%2==1:

        feeling_list.append('male_sad')

    elif item[6:-16]=='05' and int(item[18:-4])%2==0:

        feeling_list.append('female_angry')

    elif item[6:-16]=='05' and int(item[18:-4])%2==1:

        feeling_list.append('male_angry')

    elif item[6:-16]=='06' and int(item[18:-4])%2==0:

        feeling_list.append('female_fearful')

    elif item[6:-16]=='06' and int(item[18:-4])%2==1:

        feeling_list.append('male_fearful')
labels = pd.DataFrame(feeling_list)
df = pd.DataFrame(columns=['feature'])

bookmark=0

for index,y in enumerate(mylist):

    if mylist[index][6:-16]!='01' and mylist[index][6:-16]!='07' and mylist[index][6:-16]!='08' and mylist[index][:2]!='su' and mylist[index][:1]!='n' and mylist[index][:1]!='d':

        X, sample_rate = librosa.load('../input/ravdess-song-speech/RawData/'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)

        sample_rate = np.array(sample_rate)

        mfccs = np.mean(librosa.feature.mfcc(y=X, 

                                            sr=sample_rate, 

                                            n_mfcc=13),

                        axis=0)

        feature = mfccs

        #[float(i) for i in feature]

        #feature1=feature[:135]

        df.loc[bookmark] = [feature]

        bookmark=bookmark+1
df.head()
df3 = pd.DataFrame(df['feature'].values.tolist())

newdf = pd.concat([df3,labels], axis=1)

rnewdf = newdf.rename(index=str, columns={"0": "label"})

rnewdf[:5]
rnewdf=rnewdf.fillna(0)
newdf1 = np.random.rand(len(rnewdf)) < 0.8

train = rnewdf[newdf1]

test = rnewdf[~newdf1]
trainfeatures = train.iloc[:, :-1]

trainlabel = train.iloc[:, -1:]

testfeatures = test.iloc[:, :-1]

testlabel = test.iloc[:, -1:]
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder



X_train = np.array(trainfeatures)

y_train = np.array(trainlabel)

X_test = np.array(testfeatures)

y_test = np.array(testlabel)



lb = LabelEncoder()



y_train = np_utils.to_categorical(lb.fit_transform(y_train))

y_test = np_utils.to_categorical(lb.fit_transform(y_test))
x_traincnn =np.expand_dims(X_train, axis=2)

x_testcnn= np.expand_dims(X_test, axis=2)
model = Sequential()



model.add(Conv1D(256, 5,padding='same',

                 input_shape=(216,1)))

model.add(Activation('relu'))

model.add(Conv1D(128, 5,padding='same'))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(MaxPooling1D(pool_size=(8)))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Dropout(0.15))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Dropout(0.2))

model.add(Conv1D(128, 5,padding='same',))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(10))

model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=700, validation_data=(x_testcnn, y_test))
plt.plot(cnnhistory.history['loss'])

plt.plot(cnnhistory.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(cnnhistory.history['accuracy'])

plt.plot(cnnhistory.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model_name = 'Emotion_Voice_Detection_Model.h5'

save_dir = os.path.join(os.getcwd(), 'saved_models')

# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)
import json

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
# loading json and creating model

from keras.models import model_from_json

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")

print("Loaded model from disk")

 

# evaluate loaded model on test data

loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
preds = loaded_model.predict(x_testcnn, 

                         batch_size=32, 

                         verbose=1)

preds
preds1=preds.argmax(axis=1)

preds1
abc = preds1.astype(int).flatten()

predictions = (lb.inverse_transform((abc)))

preddf = pd.DataFrame({'predictedvalues': predictions})

preddf[:10]
actual=y_test.argmax(axis=1)

abc123 = actual.astype(int).flatten()

actualvalues = (lb.inverse_transform((abc123)))



actualdf = pd.DataFrame({'actualvalues': actualvalues})

actualdf[:10]
finaldf = actualdf.join(preddf)

finaldf[170:180]
finaldf.groupby('predictedvalues').count()
finaldf.to_csv('Predictions.csv', index=False)