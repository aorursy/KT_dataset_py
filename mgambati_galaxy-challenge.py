!pip install Augmentor
from string import Template

from zipfile import ZipFile

from os import path, mkdir

import pandas as pd

from shutil import copy, rmtree

import matplotlib.pyplot as plt

import numpy as np

from skimage import io

import Augmentor

from tqdm.autonotebook import tqdm

from sklearn.model_selection import train_test_split
original_training_data = pd.read_csv('/kaggle/input/galaxy-zoo-the-galaxy-challenge/44352/training_solutions_rev1.csv')



# Pandas coloca o galaxyID como float, convertemos novamente para string.

original_training_data['GalaxyID'] = original_training_data['GalaxyID'].astype(str)



columns_mapper = {

    'GalaxyID': 'GalaxyID',

    'Class7.1': 'completely_round',

    'Class7.2': 'in_between',

    'Class7.3': 'cigar_shaped',

    'Class2.1': 'on_edge',

    'Class4.1': 'has_signs_of_spiral',

    'Class3.1': 'spiral_barred',

    'Class3.2': 'spiral'

}

columns = list(columns_mapper.values())

training_df = original_training_data.rename(columns=columns_mapper)[columns]

training_df.set_index('GalaxyID', inplace=True)

training_df.head(10)
def plot_distribution(df, column):

    plt.plot(list(df[column]))

    plt.title('Distribution of data')

    plt.ylabel('% Votes')

    plt.legend([column], loc='upper left')

    plt.show()
# DataFrames of each class

completely_round_df = training_df.sort_values(by = 'completely_round', ascending= False)[0:5000]

completely_round_df['type'] = 'completely_round'

completely_round_df = completely_round_df[['type', 'completely_round']]

plot_distribution(completely_round_df, 'completely_round')
in_between_df = training_df.sort_values(by = 'in_between', ascending= False)[0:3600]

in_between_df['type'] = 'in_between'



# filters

bigger_than_completely_round = in_between_df['in_between'] > in_between_df['completely_round']

bigger_than_cigar_shaped = in_between_df['in_between'] > in_between_df['cigar_shaped']



in_between_df = in_between_df[bigger_than_completely_round & bigger_than_cigar_shaped]

in_between_df = in_between_df[['type', 'in_between']]

plot_distribution(in_between_df, 'in_between')
cigar_shaped_df = training_df.sort_values(by = 'cigar_shaped', ascending= False)[0:1550]

cigar_shaped_df['type'] = 'cigar_shaped'



# filters

bigger_than_in_between = cigar_shaped_df['cigar_shaped'] > cigar_shaped_df['in_between']

bigger_than_on_edge = cigar_shaped_df['cigar_shaped'] > cigar_shaped_df['on_edge']



cigar_shaped_df = cigar_shaped_df[bigger_than_in_between & bigger_than_on_edge]

cigar_shaped_df = cigar_shaped_df[['type', 'cigar_shaped']]

plot_distribution(cigar_shaped_df, 'cigar_shaped')
on_edge_df = training_df.sort_values(by = 'on_edge', ascending= False)[0:5100]

on_edge_df['type'] = 'on_edge'

on_edge_df = on_edge_df[['type', 'on_edge']]

plot_distribution(on_edge_df, 'on_edge')
spiral_barred_df = training_df.sort_values(by = ['spiral_barred', 'has_signs_of_spiral'], ascending= False)[0:3300]

spiral_barred_df['type'] = 'spiral_barred'

spiral_barred_df = spiral_barred_df[['type', 'spiral_barred']]

plot_distribution(spiral_barred_df, 'spiral_barred')
spiral_df = training_df.sort_values(by = ['spiral', 'has_signs_of_spiral'], ascending= False)[0:5000]

spiral_df['type'] = 'spiral'

spiral_df = spiral_df[['type', 'spiral']]

plot_distribution(spiral_df, 'spiral')
dfs = [

    completely_round_df,

    in_between_df,

    cigar_shaped_df,

    on_edge_df,

    spiral_barred_df,

    spiral_df

]





# Merge and drop and possible duplicates

merged_dfs = pd.concat(dfs, sort=False)

merged_dfs.reset_index(inplace = True)

merged_dfs.drop_duplicates(subset='GalaxyID', inplace = True)





train_merged_df, test_merged_df = train_test_split(merged_dfs, test_size=0.2)
train_merged_df.shape
test_merged_df.shape
def plot_info_set(df, name):

    countings = df.groupby('type').count().to_dict()['GalaxyID']

    labels = list(countings.keys())

    values = list(countings.values())



    fig1, ax1 = plt.subplots()

    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    ax1.axis('equal')

    fig1.suptitle(name)

    plt.tight_layout()

    plt.show()



    index = np.arange(len(labels))



    plt.bar(index, values)



    plt.xticks(index, labels, rotation=30)

    plt.show()



plot_info_set(train_merged_df, 'Train dataset')

plot_info_set(test_merged_df, 'Test dataset')

rmtree('/training', ignore_errors=True)

rmtree('/training_dataset', ignore_errors=True)

rmtree('/test', ignore_errors=True)

rmtree('/test_dataset', ignore_errors=True)
def copy_files_of_set(df, dest_folder):

    pbar = tqdm(total=df.shape[0], desc="Copying images", unit=" Images")

    if path.isdir(dest_folder) is False:

        mkdir(dest_folder)



    src_path = Template('/kaggle/input/galaxy-zoo-the-galaxy-challenge/44352/images_training_rev1/$name.jpg')



    for index, image in df.iterrows():

        dest_path = Template('/$path/$folder/').substitute(path=dest_folder, folder=image['type'])

        source_img = src_path.substitute(name=image['GalaxyID'])

        if path.isdir(dest_path) is False:

            mkdir(dest_path)

        copy(source_img, dest_path)

        pbar.update(1)

    pbar.close()

copy_files_of_set(train_merged_df, '/training')

copy_files_of_set(test_merged_df, '/test')
p = Augmentor.Pipeline("/training", "../training_dataset")

p.zoom(probability=1, max_factor=1.4, min_factor=1.4)

p.resize(probability=1, width=70, height=70)

p.process()
p = Augmentor.Pipeline("/test", "../test_dataset")

p.zoom(probability=1, max_factor=1.4, min_factor=1.4)

p.resize(probability=1, width=70, height=70)

p.process()
p = Augmentor.Pipeline('/training/', '../training_dataset/')

# p = Augmentor.Pipeline('../data/training/cigar_shaped', '../../training_augmented/cigar_shaped')

p.zoom(probability=1, max_factor=1.4, min_factor=1.4)

p.rotate_random_90(probability=0.2)

p.flip_top_bottom(probability=0.5)

p.flip_left_right(probability=0.5)

p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.5)

p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.8)

p.resize(probability=1, width=70, height=70)

p.sample(10000)
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import rmsprop, Adam

from keras import regularizers

from keras.callbacks import ModelCheckpoint
image_size = (70,70)

train_dir = '/training_dataset'

test_dir = '/test_dataset'

batch_size = 32



datagen = ImageDataGenerator()



train_generator = datagen.flow_from_directory(

    train_dir,

    class_mode='categorical',

    target_size=image_size,

    batch_size=batch_size,

)



validation_generator = datagen.flow_from_directory(

    test_dir,

    class_mode='categorical',

    target_size=image_size,

    batch_size=batch_size,

)

model = Sequential()



model.add(Conv2D(32,(3, 3), input_shape=(image_size[0], image_size[0], 3)))

model.add(Activation('relu'))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))



model.add(Conv2D(32, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))



model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.015)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(64, kernel_regularizer=regularizers.l2(0.015)))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(6))

model.add(Activation('sigmoid'))



model.compile(Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])



model.summary()
if path.isdir('/weights') is False:

    mkdir('/weights')
trains_steps = train_generator.n // train_generator.batch_size

validation_steps = validation_generator.n // validation_generator.batch_size

model_checkpoint = ModelCheckpoint('/weights/weights{epoch:08d}.h5', save_weights_only=True, period=5)



fit_result = model.fit_generator(

    train_generator,

    steps_per_epoch=trains_steps,

    validation_data =validation_generator,

    validation_steps=validation_steps,

    epochs=71,

    callbacks=[model_checkpoint]

)



model.save_weights('/weights/final_epoch.h5')
# Accuracy

plt.plot(list(range(25,80)), fit_result.history['acc'][25:80])

plt.plot(list(range(25,80)), fit_result.history['val_acc'][25:80])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Loss

plt.plot(list(range(25,80)), fit_result.history['loss'][25:80])

plt.plot(list(range(25,80)), fit_result.history['val_loss'][25:80])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()