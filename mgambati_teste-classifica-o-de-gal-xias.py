!pip install Augmentor
from string import Template

from zipfile import ZipFile

from os import path, mkdir, makedirs

import pandas as pd

from shutil import copy

import matplotlib.pyplot as plt

import numpy as np

import Augmentor

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import seaborn as sns
plt.style.use('seaborn')



COMPETITION_NAME = "galaxy-zoo-the-galaxy-challenge"

DATA_PATH = "/kaggle/input/galaxy-zoo-the-galaxy-challenge/44352/"
#  Para a classificação das galáxias, o dataset fornecido pelo galaxy challenge vem com 37 classes.

#

#  Para reduzir a quantidade classes, filtramos as classes que desejamos e e copiamos cada classe par

#  sua devida pasta. Usaremos imagens apenas com indices de respostas maiores de 90%.

# - completely-rounded: Class7.1

# - in-between: 7.2

# - cigar-shaped: Class7.3

# - on-edge: Class2.1

# - spiral-barred: Class3.1 && Class4.1

# - spiral: Class3.2 && Class4.1



#%% Loading csv and adjusting the dataframe

original_training_data = pd.read_csv(DATA_PATH + "training_solutions_rev1.csv")



# Pandas read GalaxyID has float, converts it back to string.

original_training_data["GalaxyID"] = original_training_data["GalaxyID"].astype(str)



# Better column naming

columns_mapper = {

    "GalaxyID": "GalaxyID",

    "Class7.1": "completely_round",

    "Class7.2": "in_between",

    "Class7.3": "cigar_shaped",

    "Class2.1": "on_edge",

    "Class4.1": "has_signs_of_spiral",

    "Class3.1": "spiral_barred",

    "Class3.2": "spiral",

}



columns = list(columns_mapper.values())

galaxies_df = original_training_data.rename(columns=columns_mapper)[columns]

galaxies_df.set_index("GalaxyID", inplace=True)

galaxies_df.head(10)
# ### Criar DataFrames para cada classe

#%% Simple function to plot each class data



def plot_distribution(df, column):

    print("Items: " + str(df.shape[0]))

    sns.distplot(df[column])

    plt.xlabel("% Votes")

    plt.title('Distribution - ' + column)

    plt.show()
completely_round_df = galaxies_df.sort_values(by="completely_round", ascending=False)[0:7000]

completely_round_df["type"] = "completely_round"

completely_round_df = completely_round_df[["type", "completely_round"]]



plot_distribution(completely_round_df, "completely_round")
in_between_df = galaxies_df.sort_values(by="in_between", ascending=False)[0:6000]

in_between_df["type"] = "in_between"



# filters

bigger_than_completely_round = (

    in_between_df["in_between"] > in_between_df["completely_round"]

)

bigger_than_cigar_shaped = in_between_df["in_between"] > in_between_df["cigar_shaped"]



in_between_df = in_between_df[bigger_than_completely_round & bigger_than_cigar_shaped]

in_between_df = in_between_df[["type", "in_between"]]

plot_distribution(in_between_df, "in_between")
cigar_shaped_df = galaxies_df.sort_values(by="cigar_shaped", ascending=False)[0:1550]

cigar_shaped_df["type"] = "cigar_shaped"



# filters

bigger_than_in_between = cigar_shaped_df["cigar_shaped"] > cigar_shaped_df["in_between"]

bigger_than_on_edge = cigar_shaped_df["cigar_shaped"] > cigar_shaped_df["on_edge"]



cigar_shaped_df = cigar_shaped_df[bigger_than_in_between & bigger_than_on_edge]

cigar_shaped_df = cigar_shaped_df[["type", "cigar_shaped"]]



plot_distribution(cigar_shaped_df, "cigar_shaped")
on_edge_df = galaxies_df.sort_values(by="on_edge", ascending=False)[0:5000]

on_edge_df["type"] = "on_edge"

on_edge_df = on_edge_df[["type", "on_edge"]]

plot_distribution(on_edge_df, "on_edge")
spiral_barred_df = galaxies_df.sort_values(

    by=["spiral_barred", "has_signs_of_spiral"], ascending=False

)[0:4500]



spiral_barred_filter = spiral_barred_df['spiral'] < spiral_barred_df['spiral_barred']

spiral_barred_df = spiral_barred_df[spiral_barred_filter]

spiral_barred_df["type"] = "spiral_barred"

spiral_barred_df = spiral_barred_df[["type", "spiral_barred"]]

plot_distribution(spiral_barred_df, "spiral_barred")
spiral_df = galaxies_df.sort_values(

    by=["spiral", "has_signs_of_spiral"], ascending=False

)[0:8000]

spiral_df["type"] = "spiral"

spiral_df = spiral_df[["type", "spiral"]]

plot_distribution(spiral_df, "spiral")
#%% Generate a single dataframe with all galaxies from each class

dfs = [

    completely_round_df,

    in_between_df,

    cigar_shaped_df,

    on_edge_df,

    spiral_barred_df,

    spiral_df,

]





# Merge and drop and possible duplicates

merged_dfs = pd.concat(dfs, sort=False)

merged_dfs.reset_index(inplace=True)

merged_dfs.drop_duplicates(subset="GalaxyID", inplace=True)

merged_dfs.head(5)
# Split the datafrane between train and test

train_df, validation_df = train_test_split(merged_dfs, test_size=0.2)

#%% plot distribuition

def plot_info_set(df, name):

    countings = df.groupby("type").count().to_dict()["GalaxyID"]

    labels = list(countings.keys())

    values = list(countings.values())

    index = np.arange(len(labels))

    plt.bar(index, values)

    plt.title(name)

    plt.xticks(index, labels, rotation=30)

    plt.show()





plot_info_set(train_df, "Train dataset")

plot_info_set(validation_df, "Test dataset")
ZOOM_FACTOR=1.6

DIMEN=70

FILTERED_DATA_PATH = "/data/filtered/"

DATASETS_PATH = "/data/sets/"
def copy_files_of_set(df, dataset):

    print("Copying filtered files of " + dataset)

    if path.isdir(FILTERED_DATA_PATH + dataset) is False:

        makedirs(FILTERED_DATA_PATH + dataset, exist_ok=True)



    src_path = Template(DATA_PATH + "images_training_rev1/$name.jpg")



    for index, image in df.iterrows():

        dest_path = FILTERED_DATA_PATH + dataset + '/' + image['type']

        source_img = src_path.substitute(name=image["GalaxyID"])



        if path.isdir(dest_path) is False:

            mkdir(dest_path)



        copy(source_img, dest_path)
copy_files_of_set(train_df, "training")

copy_files_of_set(validation_df, "validation")
def resize_and_zoom(dataset):

    p = Augmentor.Pipeline(FILTERED_DATA_PATH + dataset, DATASETS_PATH + dataset)

    p.zoom(probability=1, max_factor=ZOOM_FACTOR, min_factor=ZOOM_FACTOR)

    p.resize(probability=1, width=DIMEN, height=DIMEN)

    p.process()

    

def augment_set(n, dataset = ""):

    p = Augmentor.Pipeline(FILTERED_DATA_PATH + "training/" + dataset, DATASETS_PATH + "training/" + dataset)

    p.zoom(probability=1, max_factor=ZOOM_FACTOR, min_factor=ZOOM_FACTOR)

    p.rotate_random_90(probability=0.2)

    p.flip_top_bottom(probability=0.5)

    p.flip_left_right(probability=0.5)

    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.5)

    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.8)

    p.resize(probability=1, width=DIMEN, height=DIMEN)

    p.sample(n)
resize_and_zoom("training")

resize_and_zoom("validation")

# augment_set(2500, "cigar_shaped")

# augment_set(1500, "spiral_barred")

augment_set(n = 15000)
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense, Input

from keras.optimizers import rmsprop, Adam

from keras import regularizers

from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from os import path, mkdir
IMAGE_SIZE = (DIMEN, DIMEN)

INPUT_SHAPE = (DIMEN, DIMEN, 3)



BATCH_SIZE = 32

TRAIN_DIR = DATASETS_PATH + "training"

VALIDATION_DIR = DATASETS_PATH + "validation"
model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))

model.add(Activation("relu"))



model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))



model.add(Conv2D(32, (3, 3)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))





model.add(Conv2D(64, (3, 3)))

model.add(Activation("relu"))



# model.add(Conv2D(64, (3, 3)))

model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.015)))

model.add(Activation("relu"))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

# model.add(Dense(64))

model.add(Dense(64, kernel_regularizer=regularizers.l2(0.015)))

model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(6))

model.add(Activation("softmax"))



model.compile(Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])



model.summary()
datagen = ImageDataGenerator()



train_generator = datagen.flow_from_directory(

    TRAIN_DIR, class_mode="sparse", target_size=IMAGE_SIZE, batch_size=BATCH_SIZE

)



validation_generator = datagen.flow_from_directory(

    VALIDATION_DIR, class_mode="sparse", target_size=IMAGE_SIZE, batch_size=BATCH_SIZE

)
if path.isdir("/weights") is False:

    mkdir("/weights")



trains_steps = train_generator.n // train_generator.batch_size

validation_steps = validation_generator.n // validation_generator.batch_size



model_checkpoint = ModelCheckpoint(

    "/weights/weights{epoch:08d}.h5", save_weights_only=True, period=5

)



fit_result = model.fit_generator(

    train_generator,

    steps_per_epoch=trains_steps,

    validation_data=validation_generator,

    validation_steps=validation_steps,

    epochs=80,

    callbacks=[model_checkpoint],

)



model.save_weights("/weights/final_epoch.h5")
#%%

# Accuracy



plt.plot(fit_result.history["acc"])

plt.plot(fit_result.history["val_acc"])

plt.title("Model accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train", "Test"], loc="upper left")

plt.show()



# Loss

plt.plot(fit_result.history["loss"])

plt.plot(fit_result.history["val_loss"])

plt.title("Model loss")

plt.ylabel("Loss")

plt.xlabel("Epoch")

plt.legend(["Train", "Test"], loc="upper left")

plt.show()
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np

import seaborn as sns
prediction_generator = datagen.flow_from_directory(

    VALIDATION_DIR, class_mode="sparse", target_size=IMAGE_SIZE, batch_size=1, shuffle=False,

)
predict_result = model.predict_generator(prediction_generator, prediction_generator.n)

y_predicts = np.argmax(predict_result, axis=1)

classes_labels = list(prediction_generator.class_indices.keys())
print(classification_report(prediction_generator.classes, y_predicts, target_names=classes_labels))
sns.heatmap(confusion_matrix(prediction_generator.classes, y_predicts), xticklabels=classes_labels, yticklabels=classes_labels, annot=True,fmt='.5g') 