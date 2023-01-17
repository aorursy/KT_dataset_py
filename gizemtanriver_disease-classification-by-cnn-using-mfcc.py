# Load various imports 

from datetime import datetime

from os import listdir

from os.path import isfile, join



import librosa

import librosa.display



import numpy as np

import pandas as pd



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D

from keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint



from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns
mypath = "/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/"

filenames = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.wav'))] 
p_id_in_file = [] # patient IDs corresponding to each file

for name in filenames:

    p_id_in_file.append(int(name[:3]))



p_id_in_file = np.array(p_id_in_file) 
max_pad_len = 862 # to make the length of all MFCC equal



def extract_features(file_name):

    """

    This function takes in the path for an audio file as a string, loads it, and returns the MFCC

    of the audio"""

   

    try:

        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast', duration=20) 

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

        pad_width = max_pad_len - mfccs.shape[1]

        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

        

    except Exception as e:

        print("Error encountered while parsing file: ", file_name)

        return None 

     

    return mfccs
filepaths = [join(mypath, f) for f in filenames] # full paths of files
p_diag = pd.read_csv("/kaggle/input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv",header=None) # patient diagnosis file
labels = np.array([p_diag[p_diag[0] == x][1].values[0] for x in p_id_in_file]) # labels for audio files
features = [] 



# Iterate through each sound file and extract the features

for file_name in filepaths:

    data = extract_features(file_name)

    features.append(data)



print('Finished feature extraction from ', len(features), ' files')

features = np.array(features)
# plot an MFCC

plt.figure(figsize=(10, 4))

librosa.display.specshow(features[7], x_axis='time')

plt.colorbar()

plt.title('MFCC')

plt.tight_layout()

plt.show()
features = np.array(features) # convert to numpy array
# delete the very rare diseases

features1 = np.delete(features, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0) 



labels1 = np.delete(labels, np.where((labels == 'Asthma') | (labels == 'LRTI'))[0], axis=0)
# print class counts

unique_elements, counts_elements = np.unique(labels1, return_counts=True)

print(np.asarray((unique_elements, counts_elements)))
# plot class counts

y_pos = np.arange(len(unique_elements))

plt.figure(figsize=(12,8))

plt.bar(unique_elements, counts_elements, align='center', alpha=0.5)

plt.xticks(y_pos, unique_elements)

plt.ylabel('Count')

plt.xlabel('Disease')

plt.title('Disease Count in Sound Files (No Asthma or LRTI)')

plt.show()
# One-hot encode labels

le = LabelEncoder()

i_labels = le.fit_transform(labels1)

oh_labels = to_categorical(i_labels) 
# add channel dimension for CNN

features1 = np.reshape(features1, (*features1.shape,1)) 
# train test split

x_train, x_test, y_train, y_test = train_test_split(features1, oh_labels, stratify=oh_labels, 

                                                    test_size=0.2, random_state = 42)
num_rows = 40

num_columns = 862

num_channels = 1



num_labels = oh_labels.shape[1]

filter_size = 2



# Construct model 

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=filter_size,

                 input_shape=(num_rows, num_columns, num_channels), activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))



model.add(Conv2D(filters=32, kernel_size=filter_size, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))



model.add(Conv2D(filters=64, kernel_size=filter_size, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))



model.add(Conv2D(filters=128, kernel_size=filter_size, activation='relu'))

model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))



model.add(GlobalAveragePooling2D())



model.add(Dense(num_labels, activation='softmax')) 
# Compile the model

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 
# Display model architecture summary 

model.summary()



# Calculate pre-training accuracy 

score = model.evaluate(x_test, y_test, verbose=1)

accuracy = 100*score[1]



print("Pre-training accuracy: %.4f%%" % accuracy)
# train model

num_epochs = 250

num_batch_size = 128



callbacks = [

    ModelCheckpoint(

        filepath='mymodel2_{epoch:02d}.h5',

        # Path where to save the model

        # The two parameters below mean that we will overwrite

        # the current checkpoint if and only if

        # the `val_accuracy` score has improved.

        save_best_only=True,

        monitor='val_accuracy',

        verbose=1)

]

start = datetime.now()



model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs,

          validation_data=(x_test, y_test), callbacks=callbacks, verbose=1)





duration = datetime.now() - start

print("Training completed in time: ", duration)
# Evaluating the model on the training and testing set

score = model.evaluate(x_train, y_train, verbose=0)

print("Training Accuracy: ", score[1])



score = model.evaluate(x_test, y_test, verbose=0)

print("Testing Accuracy: ", score[1])
preds = model.predict(x_test) # label scores 



classpreds = np.argmax(preds, axis=1) # predicted classes 



y_testclass = np.argmax(y_test, axis=1) # true classes



n_classes=6 # number of classes
# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])
c_names = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']
# Plot ROC curves

fig, ax = plt.subplots(figsize=(16, 10))

ax.plot([0, 1], [0, 1], 'k--')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.05])

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC Curve for Each Class')

for i in range(n_classes):

    ax.plot(fpr[i], tpr[i], linewidth=3, label='ROC curve (area = %0.2f) for %s' % (roc_auc[i], c_names[i]))

ax.legend(loc="best", fontsize='x-large')

ax.grid(alpha=.4)

sns.despine()

plt.show()
# Classification Report

print(classification_report(y_testclass, classpreds, target_names=c_names))
# Confusion Matrix

print(confusion_matrix(y_testclass, classpreds))