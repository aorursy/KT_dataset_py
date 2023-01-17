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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import tensorflow as tf

tf.__version__
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator( rescale = 1.0/255,

                                    # zoom_range = 0.2,

                                    samplewise_center = True,

                                    samplewise_std_normalization = True,

                                    # vertical_flip = True,

                                    validation_split = 0.1)
train_generator = datagen.flow_from_directory( directory = '../input/video-classification/Video Classification/Training',

                                                     target_size = (224, 224),

                                                     class_mode = 'categorical',

                                                     batch_size = 128,

                                                     shuffle = True,

                                                     subset = 'training')

train_generator.class_indices
validation_generator = datagen.flow_from_directory( directory = '../input/video-classification/Video Classification/Training',

                                                     target_size = (224, 224),

                                                     class_mode = 'categorical',

                                                     batch_size = 128,

                                                     subset = 'validation')

validation_generator.class_indices
for file in train_generator.filenames:

    print(file)
from tensorflow.keras.models import Sequential

# from tensorflow.keras.applications.resnet import ResNet50

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Conv2D

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers
# base_model = ResNet50(include_top = False, weights='imagenet', input_shape = (224, 224, 3))
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)
# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
'''

# instantiating the model in the strategy scope creates the model on the TPU

with tpu_strategy.scope():

    model = tf.keras.Sequential( … ) # define your model normally

    model.compile( … )



# train model normally

model.fit(training_dataset, epochs=EPOCHS, steps_per_epoch=…)

'''
with tpu_strategy.scope():

    

    my_model = Sequential()

    my_model.add(Conv2D(64, kernel_size=4, strides=1, activation='relu', input_shape=(224, 224, 3)))

    my_model.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))

    my_model.add(Dropout(0.5))

    my_model.add(Conv2D(128, kernel_size=4, strides=1, activation='relu'))

    my_model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))

    my_model.add(Dropout(0.5))

    my_model.add(Conv2D(256, kernel_size=4, strides=1, activation='relu'))

    my_model.add(Conv2D(256, kernel_size=4, strides=2, activation='relu'))

    my_model.add(Flatten())

    my_model.add(Dropout(0.5))

    my_model.add(Dense(512, activation='relu'))

    my_model.add(Dense(102, activation='softmax'))



    my_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
'''

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(1024,activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(512,activation='relu')(x)

x = Dropout(0.5)(x)

preds = Dense(102,activation='softmax')(x)

model = Model(inputs=base_model.input,outputs=preds)

'''
# model.compile(optimizer = optimizers.Adam(learning_rate = 0.001), loss = "categorical_crossentropy", metrics = ["accuracy"])
def train_input_fn(batch_size=2):

    # Convert the inputs to a Dataset.

    dataset = tf.data.Dataset.from_tensor_slices((train_generator))

# Shuffle, repeat, and batch the examples.

    dataset = dataset.cache()

    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)

    dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)

# Return the dataset.

    return dataset
history = my_model.fit_generator(train_input_fn(), epochs = 20, validation_data = validation_generator)
history_dict = history.history
dataframe = pd.DataFrame(history_dict)

dataframe.to_csv("Normal Hstory 20 epochs.csv")
scoreSeg = model.evaluate_generator(validation_generator)

print("Accuracy = ", scoreSeg[1])

print(scoreSeg)
import tensorflow as tf

bring_model = tf.keras.models.load_model("../input/models/model_video6 (till Best)( 34 93 ).h5")
scoreSeg =  bring_model.evaluate_generator(validation_generator)

print("Accuracy = ", scoreSeg[1])

print(scoreSeg)
history = bring_model3.fit_generator(train_generator, epochs = 1, validation_data = validation_generator)
scoreSeg = bring_model.evaluate_generator(validation_generator) # , 870//128 Validation Accuracy

print("Accuracy = ", scoreSeg[1])

print(scoreSeg)
scoreSeg_test = bring_model.evaluate_generator(validation_generator_test) # , 870//128 Test Accuracy

print("Accuracy = ", scoreSeg_test[1])

print(scoreSeg_test)
bring_model.save("model_video6.h5")
datagen_test = ImageDataGenerator( rescale = 1.0/255,

                                    vertical_flip = True)
validation_generator_test = datagen_test.flow_from_directory( directory = '../input/video-classification/Video Classification/Test',

                                                     target_size = (224, 224),

                                                     class_mode = 'categorical',

                                                     batch_size = 128)

scoreSeg_test = bring_model.evaluate_generator(validation_generator_test) # , 870//128

print("Accuracy = ", scoreSeg_test[1])

print(scoreSeg_test)
scoreSeg_test = bring_model2.evaluate_generator(validation_generator_test) # , 870//128

print("Accuracy = ", scoreSeg_test[1])

print(scoreSeg_test)
scoreSeg_test = bring_model3.evaluate_generator(validation_generator_test) # , 870//128

print("Accuracy = ", scoreSeg_test[1])

print(scoreSeg_test)
scoreSeg_test = bring_model4.evaluate_generator(validation_generator_test) # , 870//128

print("Accuracy = ", scoreSeg_test[1])

print(scoreSeg_test)
value = {'ApplyEyeMakeup': 0, 'ApplyLipstick': 1, 'Archery': 2, 'BabyCrawling': 3, 'BalanceBeam': 4, 'BandMarching': 5, 'BaseballPitch': 6, 'Basketball': 7, 'BasketballDunk': 8, 'BenchPress': 9, 'Biking': 10, 'Billiards': 11, 'BlowDryHair': 12, 'BlowingCandles': 13, 'BodyWeightSquats': 14, 'Bowling': 15, 'BoxingPunchingBag': 16, 'BoxingSpeedBag': 17, 'BreastStroke': 18, 'BrushingTeeth': 19, 'CleanAndJerk': 20, 'CliffDiving': 21, 'CricketBowling': 22, 'CricketShot': 23, 'CuttingInKitchen': 24, 'Diving': 25, 'Drumming': 26, 'Fencing': 27, 'FieldHockeyPenalty': 28, 'FloorGymnastics': 29, 'FrisbeeCatch': 30, 'FrontCrawl': 31, 'GolfSwing': 32, 'Haircut': 33, 'HammerThrow': 34, 'Hammering': 35, 'HandstandPushups': 36, 'HandstandWalking': 37, 'HeadMassage': 38, 'HighJump': 39, 'HorseRace': 40, 'HorseRiding': 41, 'HulaHoop': 42, 'IceDancing': 43, 'JavelinThrow': 44, 'JugglingBalls': 45, 'JumpRope': 46, 'JumpingJack': 47, 'Kayaking': 48, 'Knitting': 49, 'LongJump': 50, 'Lunges': 51, 'MilitaryParade': 52, 'Mixing': 53, 'MoppingFloor': 54, 'Nothing': 55, 'Nunchucks': 56, 'ParallelBars': 57, 'PizzaTossing': 58, 'PlayingCello': 59, 'PlayingDaf': 60, 'PlayingDhol': 61, 'PlayingFlute': 62, 'PlayingGuitar': 63, 'PlayingPiano': 64, 'PlayingSitar': 65, 'PlayingTabla': 66, 'PlayingViolin': 67, 'PoleVault': 68, 'PommelHorse': 69, 'PullUps': 70, 'Punch': 71, 'PushUps': 72, 'Rafting': 73, 'RockClimbingIndoor': 74, 'RopeClimbing': 75, 'Rowing': 76, 'SalsaSpin': 77, 'ShavingBeard': 78, 'Shotput': 79, 'SkateBoarding': 80, 'Skiing': 81, 'Skijet': 82, 'SkyDiving': 83, 'SoccerJuggling': 84, 'SoccerPenalty': 85, 'StillRings': 86, 'SumoWrestling': 87, 'Surfing': 88, 'Swing': 89, 'TableTennisShot': 90, 'TaiChi': 91, 'TennisSwing': 92, 'ThrowDiscus': 93, 'TrampolineJumping': 94, 'Typing': 95, 'UnevenBars': 96, 'VolleyballSpiking': 97, 'WalkingWithDog': 98, 'WallPushups': 99, 'WritingOnBoard': 100, 'YoYo': 101}