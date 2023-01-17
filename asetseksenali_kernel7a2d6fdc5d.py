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
data = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')

data.loc[:, 'id'] = data.loc[:, 'id'].apply(lambda x: x + '.jpg')

data.head()
import matplotlib.pyplot as plt

from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

import PIL.Image

from sklearn.metrics import log_loss
breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier', 'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier', 'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick', 'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer', 'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound', 'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever', 'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever', 'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael', 'ibizan_hound', 'irish_setter', 'irish_terrier',  'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog', 'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland', 'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler', 'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer', 'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet', 'wire-haired_fox_terrier', 'yorkshire_terrier']
resnet = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

resnet.summary()
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(dataframe=data, x_col='id', y_col='breed', subset='training', directory='/kaggle/input/dog-breed-identification/train', target_size=(299, 299))

valid_generator = train_datagen.flow_from_dataframe(dataframe=data, x_col='id', y_col='breed', subset='validation', directory='/kaggle/input/dog-breed-identification/train', target_size=(299, 299)) 
plt.imshow(next(train_generator)[0][0])
model = Sequential()

model.add(resnet)

model.add(Flatten())

model.add(Dropout(0.25))

model.add(Dense(256, activation='tanh'))

model.add(Dense(len(breeds), activation='softmax'))

model.layers[0].trainable=False
model.compile(loss='categorical_crossentropy',

             optimizer='rmsprop',

             metrics=['accuracy'])
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size



history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=50

)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(directory='/kaggle/input/dog-breed-identification/test', target_size=(299, 299), batch_size=1)
STEP_SIZE_TEST=test_generator.n

predictions = model.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predictions.shape
len(ids)
result = pd.DataFrame(predictions, columns=train_generator.class_indices.keys())

result.index += 1

result['id'] = [filename[5:-4] for filename in test_generator.filenames]
cols = result.columns.tolist()

cols = cols[-1:] + cols[:-1]

result = result[cols]
result.head()
result.to_csv('/kaggle/working/result.csv')