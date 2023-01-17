import numpy as np 

import pandas as pd 

import os

import glob

import pickle

from sklearn.model_selection import train_test_split 

import librosa
from keras.models import Model

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Input

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Reshape

from keras.layers import Activation

from keras.layers import concatenate

from keras import optimizers

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras import backend as K

import tensorflow as tf
INPUT_FOLDER = "../input/freesound-audio-tagging-2019/"

print(os.listdir(INPUT_FOLDER))
TRAIN_CURATED_PATH = INPUT_FOLDER + "train_curated.csv"

TRAIN_NOISY_PATH = INPUT_FOLDER + "train_noisy.csv"

SAMPLE_SUBMISSION_PATH = INPUT_FOLDER + "sample_submission.csv"

TRAIN_CURATED = INPUT_FOLDER + "train_curated/"

TRAIN_NOISY = INPUT_FOLDER + "train_noisy/"

TEST = INPUT_FOLDER + "test/"



train_curated = pd.read_csv(TRAIN_CURATED_PATH)

train_noisy = pd.read_csv(TRAIN_NOISY_PATH)

sample = pd.read_csv(SAMPLE_SUBMISSION_PATH)
def one_hot(labels, src_dict):

    ar = np.zeros([len(labels), len(src_dict)])

    for i, label in enumerate(labels):

        label_list = label.split(',')

        for la in label_list:

            ar[i, src_dict[la]] = 1

    return ar
target_names = sample.columns[1:]

num_targets = len(target_names)



src_dict = {target_names[i]:i for i in range(num_targets)}

src_dict_inv = {i:target_names[i] for i in range(num_targets)}
# image size, normalized

num_freq = 128

len_div = 256
# # get normarized images (num_freq, len_div). zero padding for last cut

# X_proc_ = np.zeros([1, num_freq, len_div]) # dummy of normalized image (templete)

# y_proc_ = np.zeros([1,80]) # dummy of label (templete)

# y_proc_tmp = one_hot(train_curated['labels'], src_dict)



# file_name = train_curated['fname'].values



# for i, file in enumerate(file_name):

#     wavfile = TRAIN_CURATED + file

#     y_proc, sr = librosa.load(wavfile)

#     S = librosa.feature.melspectrogram(y_proc, sr=sr, n_mels=num_freq)

#     log_S = librosa.power_to_db(S, ref=np.max)

#     X_proc = (log_S + 80) / 40 - 1 # fit signal range (-80, 0) -> (-1, 1)

    

#     num_div = X_proc.shape[1] // len_div

#     num_pad = len_div - X_proc.shape[1] % len_div

#     redidual_amp = np.zeros([num_freq, num_pad])

#     dum = np.hstack([X_proc, redidual_amp])

#     X_proc_ = np.vstack([X_proc_, np.array(np.split(dum, num_div+1,1))])

#     for _ in range(num_div+1):

#         y_proc_ = np.vstack([y_proc_, y_proc_tmp[i]])



# X = X_proc_[1:] # del templete

# y = y_proc_[1:] # del templete

# X = X.reshape([-1, num_freq, len_div, 1])



# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
with open('../input/preprocessed-train-data-2/train_arr.pickle', 'rb') as f:

    X_train = pickle.load(f)

    y_train = pickle.load(f)
inputs = Input(shape=(num_freq,len_div,1), name='input')



dense_list = []



## Block 1

conv1 = Conv2D(4, (19, 19),activation='relu',padding='same',name='conv1')(inputs)

pool1 = MaxPooling2D((19, 19),strides=(1, 1),padding='same',name='pool1')(conv1)

norm1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm1')(pool1)

drop1 = Dropout(0.05)(norm1)



conv1_1 = Conv2D(4, (11, 11),activation='relu',padding='same',name='conv1_1')(drop1)

pool1_1 = MaxPooling2D((5, 5),strides=(5, 5),padding='same',name='pool1_1')(conv1_1)

norm1_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm1_1')(pool1_1)

drop1_1 = Dropout(0.05)(norm1_1)



flatten1 = Flatten(name='flatten1')(drop1)

dense1 = Dense(16, name='dense1')(flatten1)

act1 = Activation('relu',name='act1')(dense1)

dense_list.append(act1)



## Block 2

conv2 = Conv2D(4, (13, 13),activation='relu',padding='same',name='conv2')(inputs)

pool2 = MaxPooling2D((13, 13), strides=(1, 1), padding='same',name='pool2')(conv2)

norm2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm2')(pool2)

drop2 = Dropout(rate=0.05)(norm2)



conv2_1 = Conv2D(4, (7, 7),activation='relu',padding='same',name='conv2_1')(drop2)

pool2_1 = MaxPooling2D((7, 7), strides=(5, 5), padding='same',name='pool2_1')(conv2_1)

norm2_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001,name='norm2_1')(pool2_1)

drop2_1 = Dropout(rate=0.05)(norm2_1)



flatten2 = Flatten(name='flatten2')(drop2_1)

dense2 = Dense(16, name='dense2')(flatten2)

act2 = Activation('relu',name='act2')(dense2)

dense_list.append(act2)



## Block 3

conv3 = Conv2D(8, (11, 11), activation='relu',padding='same',name='conv3')(inputs)

pool3 = MaxPooling2D((11, 11), strides=(2, 2), padding='same',name='pool3')(conv3)

norm3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm3')(pool3)

drop3 = Dropout(rate=0.05)(norm3)



conv3_1 = Conv2D(8, (5, 5), activation='relu',padding='same',name='conv3_1')(drop3)

pool3_1 = MaxPooling2D((5, 5), strides=(2, 2), padding='same',name='pool3_1')(conv3_1)

norm3_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm3_1')(pool3_1)

drop3_1 = Dropout(rate=0.05)(norm3_1)



flatten3 = Flatten(name='flatten3')(drop3_1)

dense3 = Dense(16, name='dense3')(flatten3)

act3 = Activation('relu',name='act3')(dense3)

dense_list.append(act3)



## Block 4

conv4 = Conv2D(8, (9, 9),activation='relu',padding='same',name='conv4')(inputs)

pool4 = MaxPooling2D((9, 9), strides=(2, 2), padding='same',name='pool4')(conv4)

norm4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm4')(pool4)

drop4 = Dropout(rate=0.05)(norm4)



conv4_1 = Conv2D(8, (3, 3),activation='relu',padding='same',name='conv4_1')(drop4)

pool4_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same',name='pool4_1')(conv4_1)

norm4_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001,name='norm4_1')(pool4_1)

drop4_1 = Dropout(rate=0.05)(norm4_1)



flatten4 = Flatten(name='flatten4')(drop4_1)

dense4 = Dense(16, name='dense4')(flatten4)

act4 = Activation('relu',name='act4')(dense4)

dense_list.append(act4)



concat = concatenate(dense_list, name='concat', axis=1)



dense2 = Dense(80, name='dense_all')(concat)

pred = Activation('softmax',name='pred')(dense2)



adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)



model = Model(inputs=inputs, outputs=pred)
def tf_one_sample_positive_class_precisions(y_true, y_pred) :

    num_samples, num_classes = y_pred.shape

    

    # find true labels

    pos_class_indices = tf.where(y_true > 0) 

    

    # put rank on each element

    retrieved_classes = tf.nn.top_k(y_pred, k=num_classes).indices

    sample_range = tf.zeros(shape=tf.shape(tf.transpose(y_pred)), dtype=tf.int32)

    sample_range = tf.add(sample_range, tf.range(tf.shape(y_pred)[0], delta=1))

    sample_range = tf.transpose(sample_range)

    sample_range = tf.reshape(sample_range, (-1,num_classes*tf.shape(y_pred)[0]))

    retrieved_classes = tf.reshape(retrieved_classes, (-1,num_classes*tf.shape(y_pred)[0]))

    retrieved_class_map = tf.concat((sample_range, retrieved_classes), axis=0)

    retrieved_class_map = tf.transpose(retrieved_class_map)

    retrieved_class_map = tf.reshape(retrieved_class_map, (tf.shape(y_pred)[0], num_classes, 2))

    

    class_range = tf.zeros(shape=tf.shape(y_pred), dtype=tf.int32)

    class_range = tf.add(class_range, tf.range(num_classes, delta=1))

    

    class_rankings = tf.scatter_nd(retrieved_class_map,

                                          class_range,

                                          tf.shape(y_pred))

    

    #pick_up ranks

    num_correct_until_correct = tf.gather_nd(class_rankings, pos_class_indices)



    # add one for division for "presicion_at_hits"

    num_correct_until_correct_one = tf.add(num_correct_until_correct, 1) 

    num_correct_until_correct_one = tf.cast(num_correct_until_correct_one, tf.float32)

    

    # generate tensor [num_sample, predict_rank], 

    # top-N predicted elements have flag, N is the number of positive for each sample.

    sample_label = pos_class_indices[:, 0]   

    sample_label = tf.reshape(sample_label, (-1, 1))

    sample_label = tf.cast(sample_label, tf.int32)

    

    num_correct_until_correct = tf.reshape(num_correct_until_correct, (-1, 1))

    retrieved_class_true_position = tf.concat((sample_label, 

                                               num_correct_until_correct), axis=1)

    retrieved_pos = tf.ones(shape=tf.shape(retrieved_class_true_position)[0], dtype=tf.int32)

    retrieved_class_true = tf.scatter_nd(retrieved_class_true_position, 

                                         retrieved_pos, 

                                         tf.shape(y_pred))

    # cumulate predict_rank

    retrieved_cumulative_hits = tf.cumsum(retrieved_class_true, axis=1)



    # find positive position

    pos_ret_indices = tf.where(retrieved_class_true > 0)



    # find cumulative hits

    correct_rank = tf.gather_nd(retrieved_cumulative_hits, pos_ret_indices)  

    correct_rank = tf.cast(correct_rank, tf.float32)



    # compute presicion

    precision_at_hits = tf.truediv(correct_rank, num_correct_until_correct_one)



    return pos_class_indices, precision_at_hits



def tf_lwlrap(y_true, y_pred):

    num_samples, num_classes = y_pred.shape

    pos_class_indices, precision_at_hits = (tf_one_sample_positive_class_precisions(y_true, y_pred))

    pos_flgs = tf.cast(y_true > 0, tf.int32)

    labels_per_class = tf.reduce_sum(pos_flgs, axis=0)

    weight_per_class = tf.truediv(tf.cast(labels_per_class, tf.float32),

                                  tf.cast(tf.reduce_sum(labels_per_class), tf.float32))

    sum_precisions_by_classes = tf.zeros(shape=(num_classes), dtype=tf.float32)  

    class_label = pos_class_indices[:,1]

    sum_precisions_by_classes = tf.unsorted_segment_sum(precision_at_hits,

                                                        class_label,

                                                       num_classes)

    labels_per_class = tf.cast(labels_per_class, tf.float32)

    labels_per_class = tf.add(labels_per_class, 1e-7)

    per_class_lwlrap = tf.truediv(sum_precisions_by_classes,

                                  tf.cast(labels_per_class, tf.float32))

    out = tf.cast(tf.tensordot(per_class_lwlrap, weight_per_class, axes=1), dtype=tf.float32)

    return out
model.compile(optimizer=adam,

              loss='categorical_crossentropy',

              metrics=[tf_lwlrap])
model.summary()
datagen = ImageDataGenerator(

           rotation_range=0,

           width_shift_range=16,

           height_shift_range=0,

           shear_range=0,

           zoom_range=0,

           horizontal_flip=False,

           vertical_flip=False)
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),

                    steps_per_epoch=len(X_train) / 32, epochs=30)
# X_test_list = []



# filename = glob.glob(TEST + "*")



# for file in filename:

#     wavfile = file

#     y_proc, sr = librosa.load(wavfile)

#     S = librosa.feature.melspectrogram(y_proc, sr=sr, n_mels=num_freq)

#     log_S = librosa.power_to_db(S, ref=np.max)

#     X_proc = (log_S + 80) / 40 - 1

    

#     num_div = X_proc.shape[1] // len_div

#     num_pad = len_div - X_proc.shape[1] % len_div

#     redidual_amp = np.zeros([num_freq, num_pad])

#     dum = np.hstack([X_proc, redidual_amp])

#     X_test_list.append(np.array(np.split(dum, num_div+1,1)))

    

# with open('../input/preprocessed-test-data-2/test_arr.pickle', 'wb') as f:

#     pickle.dump(X_test_list, f)

#     pickle.dump(filename, f)

    

with open('../input/preprocessed-test-data-2/test_arr.pickle', 'rb') as f:

    X_test_list = pickle.load(f)

    filename = pickle.load(f)
pred_list = []

for X_test in X_test_list:

    pred = model.predict(X_test.reshape([-1, num_freq, len_div,1])).sum(axis=0) / len(X_test)

    pred_list.append(pred)

y_pred = np.array(pred_list)
names = []

for f in filename:

    names.append(f.split("\\")[-1])

    

se_file = pd.Series(names, name='fname')
sound_names = sample.columns[1:]

label = pd.DataFrame(y_pred, columns=sound_names)



sub_df = pd.concat([se_file, label], axis=1)
sub_df.to_csv('submission.csv', index=False)