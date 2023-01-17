# To store data
import pandas as pd

# To do linear algebra
import numpy as np

# To create plots
import matplotlib.pyplot as plt

# To create nicer plots
import seaborn as sns

# To search directories
import os

# To get progression bars
from tqdm import tqdm

# To play sound in notebooks
import IPython.display as ipd

# To create models
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPool1D, Flatten
# Load sample and ids
train = pd.read_csv('../input/train.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

# Path to files
train_path = '../input/audio_train/'
test_path = '../input/audio_test/'

print('Each file in the csv-submission has three possible concatenated labels.')
print('Sample Submission Shape:\t{}'.format(sample_submission.shape))
sample_submission.head()
print('Each file has a label and a marker weather it has been verified by a human.')
print('Train Shape:\t{}'.format(train.shape))
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_files = os.listdir(train_path)
print('Number Of Train Files:\t{}'.format(len(train_files)))

test_files = os.listdir(test_path)
print('Number Of Test Files:\t{}'.format(len(test_files)))
title = 'Distribution Of Labels'
labels_grouped = train_df.groupby(['label', 'manually_verified']).count().rename(columns={'fname':'Verified'})
labels_grouped = labels_grouped.unstack().reindex(labels_grouped.unstack().sum(axis=1).sort_values(ascending=False).index)
labels_grouped.columns = ['Unverified', 'Verified']
labels_grouped.plot(kind='barh', stacked=True, title=title, figsize=(16,9))
plt.xlabel('Count')
plt.ylabel('Label')
plt.show()
from scipy.io import wavfile
fname, label, verified = train_df.sample(1).values[0]
rate, data = wavfile.read(train_path+fname)
print(label)
print('Sampling Rate:\t{}'.format(rate))
print('Total Frames:\t{}'.format(data.shape[0]))
print(data)
n = 3
fig, axarr = plt.subplots(n, 1, figsize=(16, 2*n))
for i, (fname, label) in enumerate(train_df.sample(n)[['fname', 'label']].values):
    rate, data = wavfile.read(train_path+fname)
    axarr[i].plot(data)
    axarr[i].set_title(label)
plt.tight_layout()
plt.show()
print('Sound:\t{}'.format(train[train['fname']==fname]['label'].values[0]))
ipd.Audio(train_path+fname)
file_length = []
for file in tqdm([train_path+file for file in os.listdir(train_path)] + [test_path+file for file in os.listdir(test_path)]):
    rate, data = wavfile.read(file)
    file_length.append([len(data)/rate, rate, file])
length_df = pd.DataFrame(file_length, columns=['length', 'rate', 'file'])
length_df['data'] = 'test'
length_df.loc[:train_df.shape[0], 'data'] = 'train'

fig, axarr = plt.subplots(1, 2, figsize=(16,4))
sns.distplot(length_df[length_df['data']=='train']['length'], ax=axarr[0])
axarr[0].set_title('Train: Distribution File-Lengths')
axarr[0].set_xlabel('Seconds')
sns.distplot(length_df[length_df['data']=='test']['length'], ax=axarr[1])
axarr[1].set_title('Test: Distribution File-Lengths')
axarr[1].set_xlabel('Seconds')
plt.show()
train_df['duration'] = length_df[length_df['data']=='train']['length']

plt.figure(figsize=(16,4))
sns.violinplot(data=train_df, y='duration', x='label')
plt.title('File-Lengths Per Label')
plt.xlabel('Label')
plt.ylabel('Seconds')
plt.xticks(rotation=90)
plt.show()
# Setup variables
input_length = 44100*10 # First 10 seconds for classification
n_classes = train['label'].unique().shape[0]

# Create model
model = Sequential()
model.add(Conv1D(filters=4, kernel_size=16, activation='relu', padding='same', input_shape=(input_length, 1)))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=6, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=9, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=14, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=21, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=31, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Conv1D(filters=46, kernel_size=16, activation='relu', padding='same'))
model.add(MaxPool1D(pool_size=5))
model.add(Dropout(rate=0.1))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=n_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()
# Map files to label
file_label_dict = {fname:label for fname, label in train[['fname', 'label']].values}

example_file = '6a446a35.wav'
print('File Label "{}":\n{}'.format(example_file, file_label_dict[example_file]))


# Create vector encoded labels
labelEncoder = {}
for i, label in enumerate(train['label'].unique()):
    label_array = np.zeros(n_classes)
    label_array[i] = 1
    labelEncoder[label] = label_array

example_label = 'Cello'
print('\nEncoded Label "{}":\n{}'.format(example_label, labelEncoder[example_label]))

# Remap predictions to label
prediction_to_label = {np.argmax(array):label for label, array in labelEncoder.items()}
# Define batch generator to yield random data batches
def batchGenerator(files, batch_size):
    # Generate infinite random batches
    while True:
        # Get random files
        batch_files = np.random.choice(files, batch_size, replace=False)

        # Get labels and data
        batch_label = []
        batch_data = []
        # Combine batch
        for file in batch_files:
            # Get label and data
            label = file_label_dict[file]
            rate, data = wavfile.read(train_path+file)
            # Trim data to get uniform length
            data_uniform_length = np.zeros(input_length)
            minimum = min(input_length, data.shape[0])
            data_uniform_length[:minimum] = data[:minimum]
            # Encode label
            encoded_label = labelEncoder[label]
            # Create label and data batch
            batch_label.append(encoded_label)
            batch_data.append(data_uniform_length)
        # Format batches
        batch_label = np.array(batch_label)
        batch_data = np.array(batch_data).reshape(-1, input_length, 1)

        # Batch normalisation
        minimum, maximum = batch_data.min().astype(float), batch_data.max().astype(float)
        batch_data = (batch_data - minimum) / (maximum - minimum)

        # Yield batches for training
        yield batch_data, batch_label
# Create random mask to split files in train and validation set
train_val_split_mask  = np.zeros(train.shape[0], dtype=bool)
train_val_split_mask[:8500] = True
np.random.shuffle(train_val_split_mask)

# Get train and validation files
train_files = train['fname'][train_val_split_mask]
val_files = train['fname'][~train_val_split_mask]


# Specify train and validation generators
batch_size = 50
train_generator = batchGenerator(train_files, batch_size=batch_size)
val_generator = batchGenerator(val_files, batch_size=50)
model.fit_generator(generator=train_generator, validation_data=val_generator, validation_steps=10, epochs=20, steps_per_epoch=train.shape[0]//batch_size)
prediction = []
test_data = []
test_files = os.listdir(test_path)
for fname in tqdm(test_files):
    rate, data = wavfile.read(test_path + fname)
    # Trim data to get uniform length
    data_uniform_length = np.zeros(input_length)
    minimum = min(input_length, data.shape[0])
    data_uniform_length[:minimum] = data[:minimum]
    test_data.append(data_uniform_length)
    
    if len(test_data)==50:
        test_data = np.array(test_data).reshape(-1, input_length, 1)
        prediction.extend(model.predict(test_data))
        test_data = []
test_data = np.array(test_data).reshape(-1, input_length, 1)
prediction.extend(model.predict(test_data))

#prediction = model.predict(test_data)
prediction = np.array(prediction)
best_prediction = np.flip(prediction.argsort(), axis=1)[:, :3]

final_prediction = []
for entry in best_prediction:
    best_file_predictions = []
    for label in entry:
        best_file_predictions.append(prediction_to_label[label])
    final_prediction.append(' '.join(best_file_predictions))
final_prediction[:10]
submission = pd.DataFrame()
submission['fname'] = test_files
submission['label'] = final_prediction
submission.to_csv('Submission.csv', index=False)
submission.head()
