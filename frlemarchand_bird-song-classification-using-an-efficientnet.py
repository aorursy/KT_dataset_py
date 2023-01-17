import os

os.listdir("/kaggle/input/tensorflow230")
!pip install /kaggle/input/tensorflow230/tensorflow_estimator-2.3.0-py2.py3-none-any.whl

!pip install /kaggle/input/tensorflow230/tensorflow-2.3.0rc2-cp37-cp37m-manylinux2010_x86_64.whl
!pip list | grep tensorflow
import numpy as np

import pandas as pd

import wave

from scipy.io import wavfile

import os

import librosa

from librosa.feature import melspectrogram

import warnings

from sklearn.utils import shuffle

from sklearn.utils import class_weight

from PIL import Image

from uuid import uuid4

import sklearn

from tqdm import tqdm



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers

from tensorflow.keras import Input

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation

from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, LSTM, SimpleRNN, Conv1D, Input, BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0





import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
train_df = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
train_df = train_df.query("rating>=4")



birds_count = {}

for bird_species, count in zip(train_df.ebird_code.unique(), train_df.groupby("ebird_code")["ebird_code"].count().values):

    birds_count[bird_species] = count

most_represented_birds = [key for key,value in birds_count.items() if value == 100]



train_df = train_df.query("ebird_code in @most_represented_birds")
len(train_df.ebird_code.unique())
birds_to_recognise = sorted(shuffle(most_represented_birds)[:20])

print(birds_to_recognise)
train_df = shuffle(train_df)

train_df.head()
len(train_df)
def get_sample(filename, bird, output_folder):

    wave_data, wave_rate = librosa.load(filename)

    wave_data, _ = librosa.effects.trim(wave_data)

    #only take 5s samples and add them to the dataframe

    song_sample = []

    sample_length = 5*wave_rate

    samples_from_file = []

    #The variable below is chosen mainly to create a 216x216 image

    N_mels=216

    for idx in range(0,len(wave_data),sample_length): 

        song_sample = wave_data[idx:idx+sample_length]

        if len(song_sample)>=sample_length:

            mel = melspectrogram(song_sample, n_mels=N_mels)

            db = librosa.power_to_db(mel)

            normalised_db = sklearn.preprocessing.minmax_scale(db)

            filename = str(uuid4())+".tif"

            db_array = (np.asarray(normalised_db)*255).astype(np.uint8)

            db_image =  Image.fromarray(np.array([db_array, db_array, db_array]).T)

            db_image.save("{}{}".format(output_folder,filename))

            

            samples_from_file.append({"song_sample":"{}{}".format(output_folder,filename),

                                            "bird":bird})

    return samples_from_file
%%time

warnings.filterwarnings("ignore")

samples_df = pd.DataFrame(columns=["song_sample","bird"])



#We limit the number of audio files being sampled to 1000 in this notebook to save time

#on top of having limited the number of bird species previously

sample_limit = 1000

sample_list = []



output_folder = "/kaggle/working/melspectrogram_dataset/"

os.mkdir(output_folder)

with tqdm(total=sample_limit) as pbar:

    for idx, row in train_df[:sample_limit].iterrows():

        pbar.update(1)

        try:

            audio_file_path = "/kaggle/input/birdsong-recognition/train_audio/"

            audio_file_path += row.ebird_code

            

            if row.ebird_code in birds_to_recognise:

                sample_list += get_sample('{}/{}'.format(audio_file_path, row.filename), row.ebird_code, output_folder)

            else:

                sample_list += get_sample('{}/{}'.format(audio_file_path, row.filename), "nocall", output_folder)

        except:

            raise

            print("{} is corrupted".format(audio_file_path))

            

samples_df = pd.DataFrame(sample_list)
demo_img = Image.open(samples_df.iloc[0].song_sample)

plt.imshow(demo_img)

plt.show()
samples_df = shuffle(samples_df)

samples_df[:10]
training_percentage = 0.9

training_item_count = int(len(samples_df)*training_percentage)

validation_item_count = len(samples_df)-int(len(samples_df)*training_percentage)

training_df = samples_df[:training_item_count]

validation_df = samples_df[training_item_count:]
classes_to_predict = sorted(samples_df.bird.unique())

input_shape = (216,216, 3)

effnet_layers = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)



for layer in effnet_layers.layers:

    layer.trainable = True



dropout_dense_layer = 0.3



model = Sequential()

model.add(effnet_layers)

    

model.add(GlobalAveragePooling2D())

model.add(Dense(256, use_bias=False))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(dropout_dense_layer))



model.add(Dense(len(classes_to_predict), activation="softmax"))

    

model.summary()
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.7),

             EarlyStopping(monitor='val_loss', patience=5),

             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(loss="categorical_crossentropy", optimizer='adam')
class_weights = class_weight.compute_class_weight("balanced", classes_to_predict, samples_df.bird.values)

class_weights_dict = {i : class_weights[i] for i,label in enumerate(classes_to_predict)}
training_batch_size = 32

validation_batch_size = 32

target_size = (216,216)



train_datagen = ImageDataGenerator(

    rescale=1. / 255

)



train_generator = train_datagen.flow_from_dataframe(

    dataframe = training_df,

    x_col='song_sample',

    y_col='bird',

    directory='/',

    target_size=target_size,

    batch_size=training_batch_size,

    shuffle=True,

    class_mode='categorical')





validation_datagen = ImageDataGenerator(rescale=1. / 255)

validation_generator = validation_datagen.flow_from_dataframe(

    dataframe = validation_df,

    x_col='song_sample',

    y_col='bird',

    directory='/',

    target_size=target_size,

    shuffle=False,

    batch_size=validation_batch_size,

    class_mode='categorical')
history = model.fit(train_generator,

          epochs = 20, 

          validation_data=validation_generator,

#           class_weight=class_weights_dict,

          callbacks=callbacks)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
preds = model.predict_generator(validation_generator)

validation_df = pd.DataFrame(columns=["prediction", "groundtruth", "correct_prediction"])



for pred, groundtruth in zip(preds[:16], validation_generator.__getitem__(0)[1]):

    validation_df = validation_df.append({"prediction":classes_to_predict[np.argmax(pred)], 

                                       "groundtruth":classes_to_predict[np.argmax(groundtruth)], 

                                       "correct_prediction":np.argmax(pred)==np.argmax(groundtruth)}, ignore_index=True)

validation_df
!rm -rf /kaggle/working/melspectrogram_dataset
model.load_weights("best_model.h5")
def predict_on_melspectrogram(song_sample, sample_length):

    N_mels=216



    if len(song_sample)>=sample_length:

        mel = melspectrogram(song_sample, n_mels=N_mels)

        db = librosa.power_to_db(mel)

        normalised_db = sklearn.preprocessing.minmax_scale(db)

        db_array = (np.asarray(normalised_db)*255).astype(np.uint8)



        prediction = model.predict(np.array([np.array([db_array, db_array, db_array]).T]))

        predicted_bird = classes_to_predict[np.argmax(prediction)]

        return predicted_bird

    else:

        return "nocall"
def predict_submission(df, audio_file_path):

        

    loaded_audio_sample = []

    previous_filename = ""

    wave_data = []

    wave_rate = None

    sample_length = None

    

    for idx,row in df.iterrows():

        #I added this exception as I've heard that some files may be corrupted.

        try:

            if previous_filename == "" or previous_filename!=row.audio_id:

                filename = '{}/{}.mp3'.format(audio_file_path, row.audio_id)

                wave_data, wave_rate = librosa.load(filename)

                sample_length = 5*wave_rate

            previous_filename = row.audio_id



            #basically allows to check if we are running the examples or the test set.

            if "site" in df.columns:

                if row.site=="site_1" or row.site=="site_2":

                    song_sample = np.array(wave_data[int(row.seconds-5)*wave_rate:int(row.seconds)*wave_rate])

                elif row.site=="site_3":

                    #for now, I only take the first 5s of the samples from site_3 as they are groundtruthed at file level

                    song_sample = np.array(wave_data[0:sample_length])

            else:

                #same as the first condition but I isolated it for later and it is for the example file

                song_sample = np.array(wave_data[int(row.seconds-5)*wave_rate:int(row.seconds)*wave_rate])

            

            predicted_bird = predict_on_melspectrogram(song_sample, sample_length)

            df.at[idx,"birds"] = predicted_bird

        except:

            df.at[idx,"birds"] = "nocall"

    return df
audio_file_path = "/kaggle/input/birdsong-recognition/example_test_audio"

example_df = pd.read_csv("/kaggle/input/birdsong-recognition/example_test_audio_summary.csv")

#Ajusting the example filenames and creating the audio_id column to match with the test file.

example_df["audio_id"] = [ "BLKFR-10-CPL_20190611_093000.pt540" if filename=="BLKFR-10-CPL" else "ORANGE-7-CAP_20190606_093000.pt623" for filename in example_df["filename"]]



if os.path.exists(audio_file_path):

    example_df = predict_submission(example_df, audio_file_path)

example_df
test_file_path = "/kaggle/input/birdsong-recognition/test_audio"

test_df = pd.read_csv("/kaggle/input/birdsong-recognition/test.csv")

submission_df = pd.read_csv("/kaggle/input/birdsong-recognition/sample_submission.csv")



if os.path.exists(test_file_path):

    submission_df = predict_submission(test_df, test_file_path)



submission_df[["row_id","birds"]].to_csv('submission.csv', index=False)

submission_df.head()