# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%load_ext autoreload



%autoreload 2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dir =  "/kaggle/input/data"

train = os.listdir("/kaggle/input/data")
image1 =  os.path.join(dir + "images_001")
from glob import glob 

import pandas as pd



dataset = pd.read_csv('../input/data/Data_Entry_2017.csv')

paths = {os.path.basename(x): x for x in 

                   glob(os.path.join('..', 'input', 'data',  'images*', '*', '*.png'))}

print('Scans found:', len(paths), ', Total Headers', dataset.shape[0])

dataset['path'] = dataset['Image Index'].map(paths.get)

dataset['Cardiomegaly'] = dataset['Finding Labels'].map(lambda x: 'Cardiomegaly' in x)

dataset['Finding'] = dataset['Finding Labels'].map(lambda x: x != 'No Finding').astype(int)

dataset['Patient Age'] = np.clip(dataset['Patient Age'], 5, 100)

label_freq = dataset['Finding Labels'].apply(lambda s: str(s).split('|')).explode().value_counts().sort_values(ascending=False)

rare = list(label_freq[label_freq<10].index)

dataset['labels'] = dataset['Finding Labels'].apply(lambda s: [l for l in str(s).split('|') if l not in rare])



#dataset['Finding Labels'] = dataset['Finding Labels'].map(lambda x: x.replace('No Finding', 'n'))

from itertools import chain

all_labels = np.unique(list(chain(*dataset['Finding Labels'].map(lambda x: x.split('|')).tolist())))

all_labels = [x for x in all_labels if len(x)>0]

print('All Labels ({}): {}'.format(len(all_labels), all_labels))

for c_label in all_labels:

    if len(c_label)>1: # leave out empty labels

        dataset[c_label] = dataset['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

#dataset.sample(3)

dataset['disease_vec'] = dataset.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
data = dataset[['Image Index' , 'Patient ID', 'path', 'Finding' , 'labels' , 'disease_vec']]
noFind = data[data['Finding']==0]['Image Index'].tolist()

Find = data[data['Finding']==1]['Image Index'].tolist()

noFind_path = data[data['Finding']==0]['path'].tolist()

Find_path = data[data['Finding']==1]['path'].tolist()
path = (glob(os.path.join('..' , 'input' , 'data' , 'images*' , '*' , '*' )))
# #noFind_paths = []

# #for i in noFind[:10000]:

#     destination = (glob(os.path.join('..' , 'input' , 'data' , 'images*' , '*' , i )))

#     noFind_paths.append(destination)
# Find_paths = []

# for i in Find:

#     destination = (glob(os.path.join('..' , 'input' , 'data' , 'images*' , '*' , i )))

#     Find_paths.append(destination)
final_paths = Find_path + noFind_path[:10000]



final_data = data[data['Finding']==1]

temp= data[data['Finding']==0][:10000]





final_data.append(temp)

final_paths = final_data['path'].tolist()
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

train_df, valid_df , train_y , valid_y = train_test_split(final_paths, 

                                                          final_data['disease_vec'].tolist(),

                                   test_size = 0.6, 

                                   random_state = 2000,

                                   #stratify = data['labels'].map(lambda x: x[:4]

                                     )



valid_df, test_df , valid_y , test_y = train_test_split(valid_df, 

                                                        valid_y,

                                   test_size = 0.5, 

                                   random_state = 2000)







def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):

    base_dir = os.path.dirname(in_df[path_col].values[0])

    print('## Ignore next message from keras, values are replaced anyways')

    df_gen = img_data_gen.flow_from_directory(base_dir, 

                                     class_mode = 'sparse',

                                    **dflow_args)

    df_gen.filenames = in_df[path_col].values

    df_gen.classes = np.stack(in_df[y_col].values)

    df_gen.samples = in_df.shape[0]

    df_gen.n = in_df.shape[0]

    df_gen._set_index_array()

    df_gen.directory = '' # since we have the full path

    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))

    return df_gen







SHUFFLE_BUFFER_SIZE = 1024

BATCH_SIZE = 256 

AUTOTUNE = tf.data.experimental.AUTOTUNE 

CHANNELS = 3

IMG_SIZE = 128



def parse_function(filename, label):

    image_string = tf.io.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)

    image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])

    image_normalized = image_resized / 255.0

    return image_normalized, label







def create_dataset(filenames, labels, is_training=True):



    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    

    if is_training == True:

        dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    

    return dataset



LR = 1e-5 

EPOCHS = 1
#train_df
import tensorflow as tf

model = tf.keras.models.Sequential([

                                               tf.keras.layers.Conv2D(16 , (3,3) , activation='relu', input_shape =(128, 128 ,3)),

                                               tf.keras.layers.MaxPool2D(2,2),

                                               tf.keras.layers.Conv2D(16 , (3,3) , activation='relu'),

                                               tf.keras.layers.MaxPool2D(2,2),  

                                               tf.keras.layers.Flatten(),

                                               tf.keras.layers.Dense(128 , activation = 'relu'),

                                               tf.keras.layers.Dense(32 , activation = 'relu'),

                                               tf.keras.layers.Dense(15 , activation = 'sigmoid')

                                               ])
train_ds = create_dataset(train_df, train_y)

val_ds = create_dataset(valid_df, valid_y)

test_ds = create_dataset(_df, valid_y)

@tf.function

def macro_soft_f1(y, y_hat):

    y = tf.cast(y, tf.float32)

    y_hat = tf.cast(y_hat, tf.float32)

    tp = tf.reduce_sum(y_hat * y, axis=0)

    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)

    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)

    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)

    cost = 1 - soft_f1

    macro_cost = tf.reduce_mean(cost)

    return macro_cost

@tf.function

def macro_f1(y, y_hat, thresh=0.5):

    y_pred = tf.cast(tf.greater(y_hat, thresh), tf.float32)

    tp = tf.cast(tf.math.count_nonzero(y_pred * y, axis=0), tf.float32)

    fp = tf.cast(tf.math.count_nonzero(y_pred * (1 - y), axis=0), tf.float32)

    fn = tf.cast(tf.math.count_nonzero((1 - y_pred) * y, axis=0), tf.float32)

    f1 = 2*tp / (2*tp + fn + fp + 1e-16)

    macro_f1 = tf.reduce_mean(f1)

    return macro_f1
model.compile(

  optimizer=tf.keras.optimizers.Adam(learning_rate=LR),

  loss=macro_soft_f1,

  metrics=[macro_f1])
# import matplotlib.pyplot as plt

# import matplotlib.image as mping

# img = mping.imread(final_paths[0])

# plt.imshow(img)
history = model.fit(train_ds,

                    epochs=EPOCHS,

                    validation_data=create_dataset(valid_df , valid_y))
model.predict(test_ds[:10])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer = RMSprop(lr= 0.001) ,

             loss=macro_soft_f1,

            metrics=[macro_f1])
#print(datagen_valid)