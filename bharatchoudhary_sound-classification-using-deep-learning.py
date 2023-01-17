
# Load various imports 
import pandas as pd
import numpy as np
import os
import librosa
import glob 
import numpy as np
import skimage
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
metadata = pd.read_csv('UrbanSound8K.csv')
metadata.head()
metadata[metadata['class']=='street_music'][['slice_file_name','fold']].iloc[0,:]
print(metadata['class'].value_counts())
def plot_spectrogram(signal, name):
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    plt.figure(figsize=(25, 8))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for {name}")
    plt.xlabel("Time")
    plt.show()
street_music_file = metadata[metadata['class']=='street_music'][['slice_file_name','fold']].iloc[0,:]
drilling_file = metadata[metadata['class']=='drilling'][['slice_file_name','fold']].iloc[0,:]
engine_idling_file = metadata[metadata['class']=='engine_idling'][['slice_file_name','fold']].iloc[0,:]
children_playing_file = metadata[metadata['class']=='children_playing'][['slice_file_name','fold']].iloc[0,:]
dog_bark_file = metadata[metadata['class']=='dog_bark'][['slice_file_name','fold']].iloc[0,:]
jackhammer_file = metadata[metadata['class']=='jackhammer'][['slice_file_name','fold']].iloc[0,:]
air_conditioner_file = metadata[metadata['class']=='air_conditioner'][['slice_file_name','fold']].iloc[0,:]
siren_file = metadata[metadata['class']=='siren'][['slice_file_name','fold']].iloc[0,:]
car_horn_file = metadata[metadata['class']=='car_horn'][['slice_file_name','fold']].iloc[0,:]
gun_shot_file = metadata[metadata['class']=='gun_shot'][['slice_file_name','fold']].iloc[0,:]
# load sounds
music, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(street_music_file[1])+'/',str(street_music_file[0])))
drilling, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(drilling_file[1])+'/',str(drilling_file[0])))
engine_idling, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(engine_idling_file[1])+'/',str(engine_idling_file[0])))
children_playing, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(children_playing_file[1])+'/',str(children_playing_file[0])))
dog_bark, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(dog_bark_file[1])+'/',str(dog_bark_file[0])))
jackhammer, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(jackhammer_file[1])+'/',str(jackhammer_file[0])))
air_conditioner, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(air_conditioner_file[1])+'/',str(air_conditioner_file[0])))
siren, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(siren_file[1])+'/',str(siren_file[0])))
car_horn, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(car_horn_file[1])+'/',str(car_horn_file[0])))
gun_shot, _ = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(gun_shot_file[1])+'/',str(gun_shot_file[0])))
# Street Music

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(street_music_file[1])+'/',str(street_music_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(street_music_file[1])+'/',str(street_music_file[0])))
plot_spectrogram(music, "music")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(street_music_file[1])+'/',str(street_music_file[0])))
# Drilling

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(drilling_file[1])+'/',str(drilling_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(drilling_file[1])+'/',str(drilling_file[0])))
plot_spectrogram(drilling, "drilling")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(drilling_file[1])+'/',str(drilling_file[0])))
# Enngine Idling

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(engine_idling_file[1])+'/',str(engine_idling_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(engine_idling_file[1])+'/',str(engine_idling_file[0])))
plot_spectrogram(engine_idling, "engine_idling")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(engine_idling_file[1])+'/',str(engine_idling_file[0])))
# Children Playing

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(children_playing_file[1])+'/',str(children_playing_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(children_playing_file[1])+'/',str(children_playing_file[0])))
plot_spectrogram(children_playing, "children_playing")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(children_playing_file[1])+'/',str(children_playing_file[0])))
# Dog Barking

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(dog_bark_file[1])+'/',str(dog_bark_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(dog_bark_file[1])+'/',str(dog_bark_file[0])))
plot_spectrogram(dog_bark, "dog_barking")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(dog_bark_file[1])+'/',str(dog_bark_file[0])))
# Jackhammer

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(jackhammer_file[1])+'/',str(jackhammer_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(jackhammer_file[1])+'/',str(jackhammer_file[0])))
plot_spectrogram(jackhammer, "jackhammer")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(jackhammer_file[1])+'/',str(jackhammer_file[0])))
# Air Conditioner

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(air_conditioner_file[1])+'/',str(air_conditioner_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(air_conditioner_file[1])+'/',str(air_conditioner_file[0])))
plot_spectrogram(air_conditioner, "airconditioner")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(air_conditioner_file[1])+'/',str(air_conditioner_file[0])))
# Siren

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(siren_file[1])+'/',str(siren_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(siren_file[1])+'/',str(siren_file[0])))
plot_spectrogram(siren, "siren")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(siren_file[1])+'/',str(siren_file[0])))
# Car_Horn

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(car_horn_file[1])+'/',str(car_horn_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(car_horn_file[1])+'/',str(car_horn_file[0])))
plot_spectrogram(car_horn, "car_horn")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(car_horn_file[1])+'/',str(car_horn_file[0])))
# Gun_Shot

data,sample_rate = librosa.load(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(gun_shot_file[1])+'/',str(gun_shot_file[0])))
_ = librosa.display.waveplot(data,sr=sample_rate)
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(gun_shot_file[1])+'/',str(gun_shot_file[0])))
plot_spectrogram(gun_shot, "gun_shot")
ipd.Audio(os.path.abspath('folder/')+'/'+os.path.join('fold'+str(gun_shot_file[1])+'/',str(gun_shot_file[0])))


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
     
    return mfccsscaled
# Set the path to the full UrbanSound dataset 
fulldatasetpath = '/home/ubuntu/deep_learning/folder'

metadata = pd.read_csv(fulldatasetpath + '/UrbanSound8K.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    
    file_name = os.path.abspath('folder/')+'/'+os.path.join('fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    
    class_label = row["class"]
    data = extract_features(file_name)
    
    features.append([data, class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)
x_test.shape
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

num_rows = 40
num_columns = 174
num_channels = 1

#x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
#x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = yy.shape[1]
filter_size = 2


# Construct model 
model = Sequential()

model.add(Dense(256, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))
# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()


from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 145
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

history =  model.fit(x_train, y_train, batch_size=num_batch_size,epochs=145, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])
import librosa 
import numpy as np 

def extract_feature(file_name):
   
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None, None

    return np.array([mfccsscaled])
def print_prediction(file_name):
    prediction_feature = extract_feature(file_name) 

    predicted_vector = model.predict_classes(prediction_feature)
    predicted_class = le.inverse_transform(predicted_vector) 
    print("The predicted class is:", predicted_class[0], '\n') 

    predicted_proba_vector = model.predict_proba(prediction_feature) 
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)): 
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f') )
# Class: Air Conditioner

filename = 'folder'+'/'+os.path.join('fold'+str(street_music_file[1])+'/'+str(street_music_file[0]))

print_prediction(filename)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import seaborn as sns
y_pred = model.predict(x_test)

matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
plt.figure(figsize=(10,8))
sns.heatmap(matrix, annot=True, cmap='Blues')
plt.figure(figsize=(10,8))
sns.heatmap(matrix/np.sum(matrix), annot=True, 
            fmt='.2%', cmap='Blues')
print('Classification Report')
target_names = list(le.classes_)
print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=target_names))


