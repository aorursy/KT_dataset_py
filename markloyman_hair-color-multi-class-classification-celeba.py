import os

import numpy as np

import random

import tensorflow.compat.v1 as tf

import keras.backend as K



# set a seed for reproducible results

os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(1234)

random.seed(1234)

tf.set_random_seed(1234)



from typing import List, Tuple, Callable

import pandas as pd

import matplotlib.pyplot as plt



from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import ModelCheckpoint

from keras.layers import Dropout, Dense, GlobalAveragePooling2D

from keras.models import Model

from keras.optimizers import SGD

from keras.utils import Sequence



from celeba_utils import Config  # struct with configurations 

from celeba_utils import AttrColumns, Paths

from celeba_utils import PartitionType, DataPartition, load_partition_table_with_attributes

from celeba_utils import load_image_set





# Configure current run



Config.TRAINING_SAMPLES = 1984

Config.VALIDATION_SAMPLES = 1984

Config.BATCH_SIZE = 32

Config.NUM_EPOCHS = 15

def _create_partition(df: pd.DataFrame, sample_strategy: Callable) -> DataPartition:

    _training = sample_strategy(

        df, PartitionType.TRAINING, Config.TRAINING_SAMPLES)



    _validation = sample_strategy(

        df, PartitionType.VALIDATION, Config.VALIDATION_SAMPLES)



    _test = sample_strategy(

        df, PartitionType.TEST, Config.TEST_SAMPLES)



    return DataPartition(

        training=_training[AttrColumns.ID.value].values,

        validation=_validation[AttrColumns.ID.value].values,

        test=_test[AttrColumns.ID.value].values

    )





def _sample_random_partition(df: pd.DataFrame, partition: PartitionType, num_samples: int) -> pd.DataFrame:

    filtered = df[df['partition'] == partition.value]

    sampled = filtered.sample(num_samples)

    return sampled





def create_random_partition() -> DataPartition:

    df = pd.read_csv(Paths.DATA_PARTITION)

    return _create_partition(df, _sample_random_partition)





def trace_partition(partition: DataPartition):

    print(f"Training includes {len(partition.training)} images: {partition.training[:10]}...")

    print(f"Validation includes {len(partition.validation)} images: {partition.validation[:10]}...")

    print(f"Test includes {len(partition.test)} images: {partition.test[:10]}...")

    

    

random_partition = create_random_partition()

trace_partition(random_partition)
print(f"Readding attributes table from {Paths.ATTRIBUTES}")

df_attr = pd.read_csv(Paths.ATTRIBUTES)

df_attr.set_index(AttrColumns.ID.value, inplace=True)



# sampling relevant columns

columns = [

    AttrColumns.BALD,

    AttrColumns.BLACK_HAIR,

    AttrColumns.BLOND_HAIR,

    AttrColumns.BROWN_HAIR,

    AttrColumns.GRAY_HAIR,

]        

df_attr = df_attr[[c.value for c in columns]]

df_attr[df_attr <= 0] =  0



print("First 5 labels (one hot encoding)")

print(df_attr.head(5))



print("Distribution of attributes")

print(df_attr.sum())





def trace_attributes_per_sample_count(df, title: str):

    attributes_per_sample = np.bincount(df.sum(axis=1))

    print(title + '=>')

    print(f"\t{attributes_per_sample[0]} samples don't have any hair-color attributes")

    print(f"\t{attributes_per_sample[1]} samples have exactly one hair-color attribute")

    print(f"\t{sum(attributes_per_sample[2:])} samples have more than one hair-color attribute")



    

trace_attributes_per_sample_count(df_attr, "Total attribute count per sample")
def _sample_non_zero_partition(df: pd.DataFrame, partition: PartitionType, num_samples: int) -> pd.DataFrame:

    """

    At least one column (attribute) must be positive

    """

    filtered = df[df['partition'] == partition.value]

    values = filtered.values[:, 2:]  # ignore first 2 columns: image_id and partition

    values[values <= 0] = 0  # attribute values in CelebA are in {-1, 1}

    filtered = filtered[values.sum(axis=1) > 0]

    sampled = filtered.sample(num_samples)

    return sampled





def _sample_mutualy_exclusive_partition(df: pd.DataFrame, partition: PartitionType, num_samples: int) -> pd.DataFrame:

    """

    Columns (attributes) are mutually exclusive (only one is allowed to be possitive)

    """

    filtered = df[df['partition'] == partition.value]

    values = filtered.values[:, 2:]

    values[values <= 0] = 0

    filtered = filtered[values.sum(axis=1) == 1]

    sampled = filtered.sample(num_samples)

    return sampled





def create_hair_color_partition(sample_strategy: Callable) -> DataPartition:

    attrs = [

        AttrColumns.BALD,

        AttrColumns.BLACK_HAIR,

        AttrColumns.BLOND_HAIR,

        AttrColumns.BROWN_HAIR,

        AttrColumns.GRAY_HAIR,

    ]

    df = load_partition_table_with_attributes([c.value for c in attrs])

    return _create_partition(df, sample_strategy)





non_zero_partition = create_hair_color_partition(_sample_non_zero_partition)

trace_attributes_per_sample_count(df_attr.loc[non_zero_partition.training], "non-zero attribute count (training)")

trace_attributes_per_sample_count(df_attr.loc[non_zero_partition.validation], "non-zero attribute count (validation)")



mutually_exclusive_partition = create_hair_color_partition(_sample_mutualy_exclusive_partition)

trace_attributes_per_sample_count(df_attr.loc[mutually_exclusive_partition.training], "mutually-exclusive attribute count (training)")

trace_attributes_per_sample_count(df_attr.loc[mutually_exclusive_partition.validation], "mutually-exclusive attribute count (validation)")

def _generate_multi_class_labels(columns: List[AttrColumns], df_attr: pd.DataFrame) -> np.ndarray:

    attr = df_attr[[c.value for c in columns]]

    labels = np.array(attr.values)

    labels[labels <= 0] = 0

    # we want to keep this method independant of our sampling method

    # therefore, regardless of sampling stretagy, we want our label generation method to work with 

    # images that have no attributes or more than a single attribute

    num_of_attributes = labels.sum(axis=1)

    labels = np.where(

        (num_of_attributes != 0).reshape(-1, 1),

        labels / num_of_attributes.reshape(-1, 1),

        1 / len(columns))

    return labels

        



class MultiClassSequence(Sequence):

    def __init__(self, columns: List[AttrColumns], image_ids: List[str]):

        df_attr = pd.read_csv(Paths.ATTRIBUTES)

        df_attr.set_index(AttrColumns.ID.value, inplace=True)

               

        self.labels = _generate_multi_class_labels(

            columns,

            df_attr.loc[image_ids]

        )

        self.image_ids = image_ids

        self.batch_size = Config.BATCH_SIZE



    def __len__(self):

        return len(self.image_ids) // self.batch_size



    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:

        image_ids = self.image_ids[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = load_image_set(image_ids)

        labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        return images, labels



    

class HairColorSequence(MultiClassSequence):

    def __init__(self, image_ids: List[str]):

        hair_color_columns = [

            AttrColumns.BALD,

            AttrColumns.BLACK_HAIR,

            AttrColumns.BLOND_HAIR,

            AttrColumns.BROWN_HAIR,

            AttrColumns.GRAY_HAIR,

        ]

        super().__init__(hair_color_columns, image_ids)       

def build_model(num_classes) -> Model:

    inc_model = InceptionV3(

        weights=str(Paths.MODEL_WEIGHTS),

        include_top=False,

        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 3))

    

    x = GlobalAveragePooling2D()(inc_model.output)

    x = Dense(1024, activation="relu")(x)

    x = Dropout(0.5)(x)

    x = Dense(512, activation="relu")(x)

    x = Dense(num_classes, activation="softmax")(x)    



    model = Model(inputs=inc_model.input, outputs=x)



    for layer in model.layers[:52]:

        layer.trainable = False



    model.compile(

        optimizer=SGD(lr=0.0001, momentum=0.9), 

        loss='categorical_crossentropy', 

        metrics=['accuracy'])



    return model

# Rןun non-zero configuration: 

#   data sampling includes all rows where there is atleast one positive attribute

#   (include rows with multiple positive color attributes)



checkpoint = ModelCheckpoint(

    filepath='weights.best.non-zero.hdf5',

    verbose=1,

    save_best_only=True)



hist_non_zero = build_model(num_classes=5).fit_generator(

    HairColorSequence(non_zero_partition.training),

    validation_data=HairColorSequence(mutually_exclusive_partition.validation),

    epochs=Config.NUM_EPOCHS,

    callbacks=[checkpoint],

    verbose=1

)

# Rןun mutually-exclusive configuration: 

#   data sampling includes all rows where there is ONLY one positive attribute

#   (ignore rows with multiple positive color attributes)



checkpoint = ModelCheckpoint(

    filepath='weights.best.mutually_exclusive.hdf5',

    verbose=1,

    save_best_only=True)



hist_mutually_exclusive =  build_model(num_classes=5).fit_generator(

    HairColorSequence(mutually_exclusive_partition.training),

    validation_data=HairColorSequence(mutually_exclusive_partition.validation),

    epochs=Config.NUM_EPOCHS,

    callbacks=[checkpoint],

    verbose=2

)



plt.plot(range(Config.NUM_EPOCHS), hist_non_zero.history['val_accuracy'])

plt.plot(range(Config.NUM_EPOCHS), hist_mutually_exclusive.history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Val Accuracy')

plt.legend(['non_zero', 'mutually_exclusive'])
def _sample_mutually_exclusive_partition_with_extra_multi_attributes(df: pd.DataFrame, partition: PartitionType, num_samples: int, extention_factor = 5) -> pd.DataFrame:    

    filtered = df[df['partition'] == partition.value]

    values = filtered.values[:, 2:]

    values[values <= 0] = 0

    

    mutually_exlusize = filtered[values.sum(axis=1) == 1]

    mutually_exlusize = mutually_exlusize.sample(num_samples)

    

    multiple_attributes = filtered[values.sum(axis=1) > 1]

    multiple_attributes = multiple_attributes.sample(num_samples // extention_factor)

    

    sampled = pd.concat([mutually_exlusize, multiple_attributes]) 

    sampled = sampled.sample(frac=1.)  # reshuffle

    

    return  sampled





partition_with_extra = create_hair_color_partition(_sample_mutually_exclusive_partition_with_extra_multi_attributes)

trace_attributes_per_sample_count(df_attr.loc[partition_with_extra.training], "extended partition attribute count (training)")

trace_attributes_per_sample_count(df_attr.loc[partition_with_extra.validation], "extended partition attribute count (validation)")

checkpoint = ModelCheckpoint(

    filepath='weights.best.extended.hdf5',

    verbose=1,

    save_best_only=True)



hist_extended =  build_model(num_classes=5).fit_generator(

    HairColorSequence(partition_with_extra.training),

    validation_data=HairColorSequence(mutually_exclusive_partition.validation),

    epochs=Config.NUM_EPOCHS,

    callbacks=[checkpoint],

    verbose=2

)

plt.plot(range(Config.NUM_EPOCHS), hist_non_zero.history['val_accuracy'])

plt.plot(range(Config.NUM_EPOCHS), hist_mutually_exclusive.history['val_accuracy'])

plt.plot(range(Config.NUM_EPOCHS), hist_extended.history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Val Accuracy')

plt.legend(['non_zero', 'mutually_exclusive', 'extended'])