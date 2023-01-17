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
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
people = pd.read_csv('/kaggle/input/predicting-red-hat-business-value/people.csv.zip')
activites_train = pd.read_csv('/kaggle/input/predicting-red-hat-business-value/act_train.csv.zip')
activites_test = pd.read_csv('/kaggle/input/predicting-red-hat-business-value/act_test.csv.zip')
def merge_sets(people, activities):
    return activities.join(people.set_index('people_id'), on='people_id', lsuffix='_activity', rsuffix='_person')

train = merge_sets(people, activites_train)
test = merge_sets(people, activites_test)

train.shape, test.shape
submission_ids = test.activity_id
train.isnull().sum()
def preprocess(dataset):
    dataset = dataset.drop(['date_person', 'date_activity', 'people_id', 'activity_id'], axis=1) # 
    dataset = dataset.fillna('missing')
    return dataset

train = preprocess(train)
test = preprocess(test)
train.dtypes # Quick look at the different column types before checking categories
categorical_data = train.select_dtypes(include=['object'])
categorical_data.nunique()
column_names = categorical_data.columns
print("CATEGORICAL COLUMNS : \n" + str(column_names))

embed_column_indicators = categorical_data.nunique() > 12
onehot_column_indicators = categorical_data.nunique() <= 12

EMBED_COLUMN_NAMES = column_names[embed_column_indicators]
ONEHOT_COLUMN_NAMES = column_names[onehot_column_indicators]
NUM_COLUMN_NAMES = train.select_dtypes(exclude=['object']).columns

print("\nEMBED COLUMNS : \n" + str(EMBED_COLUMN_NAMES))
print("\nONEHOT COLUMNS : \n" + str(ONEHOT_COLUMN_NAMES))
print("\nNUMERICAL COLUMNS : \n" + str(NUM_COLUMN_NAMES))
EMBED_COLUMN_NAMES = ['char_1_activity', 'char_2_activity', 'char_8_activity',
       'char_9_activity', 'char_10_activity', 'group_1', 'char_3_person',
       'char_4_person', 'char_7_person']
ONEHOT_COLUMN_NAMES = ['activity_category', 'char_3_activity', 'char_4_activity',
       'char_5_activity', 'char_6_activity', 'char_7_activity',
       'char_1_person', 'char_2_person', 'char_5_person', 'char_6_person',
       'char_8_person', 'char_9_person']
NUM_COLUMN_NAMES = ['char_10_person', 'char_11', 'char_12', 'char_13', 'char_14',
       'char_15', 'char_16', 'char_17', 'char_18', 'char_19', 'char_20',
       'char_21', 'char_22', 'char_23', 'char_24', 'char_25', 'char_26',
       'char_27', 'char_28', 'char_29', 'char_30', 'char_31', 'char_32',
       'char_33', 'char_34', 'char_35', 'char_36', 'char_37', 'char_38']
np.amax(train[NUM_COLUMN_NAMES], axis=0)
train[NUM_COLUMN_NAMES].astype('float32').hist(figsize=(32,32))
trainset, valset = train_test_split(train, test_size=0.2)
testset = test
print(len(trainset), 'train examples')
print(len(valset), 'validation examples')
print(len(testset), 'test examples')
print(type(trainset),type(valset),type(testset))
def dataframe_to_dataset(dataframe, shuffle=False, batch_size=32, train=False, buffer_size=50000):
    dataframe = dataframe.copy()
    if train:
        labels = dataframe.pop('outcome')
    else:
        labels = tf.ones(shape=dataframe.shape[0]) * -1 # Will be ignored at test time
        
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        buffer_size = buffer_size if buffer_size < dataframe.shape[0] else dataframe.shape[0]
        dataset = dataset.shuffle(buffer_size, seed=42)
        
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
train_dataset = dataframe_to_dataset(trainset, train=True, shuffle=True, batch_size=4096)
val_dataset = dataframe_to_dataset(valset, train=True, batch_size=4096)
test_dataset = dataframe_to_dataset(testset, batch_size=4096)
feature_columns = []

for column_name in NUM_COLUMN_NAMES:
    feature_columns.append(tf.feature_column.numeric_column(key=column_name, dtype=tf.float32, default_value=-1))
    
for column_name in ONEHOT_COLUMN_NAMES:
    vocabulary = train[column_name].unique()
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                          key=column_name, vocabulary_list=vocabulary, default_value=-1)
    
    onehot_column = tf.feature_column.indicator_column(categorical_column)
    feature_columns.append(onehot_column)
    
for column_name in EMBED_COLUMN_NAMES:
    vocabulary = train[column_name].unique()
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                          key=column_name, vocabulary_list=vocabulary, default_value=-1)
    
    embedding_dimensions = 16 if len(vocabulary) < 500 else 256
    embed_column = tf.feature_column.embedding_column(categorical_column, dimension=embedding_dimensions)
    feature_columns.append(embed_column)
print(feature_columns[0],'\n')
print(feature_columns[35],'\n')
print(feature_columns[48],'\n')
processing_layer = keras.layers.DenseFeatures(feature_columns)
model = keras.models.Sequential()
model.add(processing_layer)
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(16, kernel_initializer='he_normal'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.01)))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Nadam(0.01),
              metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
reduce_lr_plateau = keras.callbacks.ReduceLROnPlateau(patience=3)
history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=[early_stopping, reduce_lr_plateau], steps_per_epoch=20, validation_steps=20)










import matplotlib.pyplot as plt

plt.plot(history.epoch, history.history["lr"], "bo-")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate", color='b')
plt.tick_params('y', colors='b')
plt.gca().set_xlim(0, 10 - 1)
plt.grid(True)

ax2 = plt.gca().twinx()
ax2.plot(history.epoch, history.history["val_loss"], "r^-")
ax2.set_ylabel('Validation Loss', color='r')
ax2.tick_params('y', colors='r')

plt.title("Reduce LR on Plateau", fontsize=14)
plt.show()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0.7, 1)
plt.show()
val_preds = model.predict(val_dataset, verbose=True)
from sklearn.metrics import roc_auc_score
y_true = list(val_dataset.map(lambda x,y: y).unbatch().as_numpy_iterator())
roc_auc_score(y_true, val_preds)
test_preds = model.predict(test_dataset, verbose=True)
np.mean(test_preds), test_preds[0], test_preds
ids = np.array(submission_ids).reshape((-1,)) # make 1-dimensional
preds = test_preds.reshape((-1,)) # make 1-dimensional
print(ids.shape, preds.shape)

my_submission = pd.DataFrame({'activity_id': ids, 'outcome': preds})

my_submission.to_csv('REDHAT_submission_2.csv', index=False)
