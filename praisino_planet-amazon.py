import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import pathlib

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import cv2

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD

from sklearn.utils import shuffle
from sklearn.metrics import fbeta_score
# set path
path = '../input/planets-dataset/'
os.chdir(path)   

for dirname, _, filenames in os.walk('planet'):
    for filename in filenames:
        os.path.join(dirname, filename)
!ls
# load train and test datasets
train_data = pd.read_csv("planet/planet/train_classes.csv")
test_data = pd.read_csv("planet/planet/sample_submission.csv")

# preview train dataset
train_data.head()
# Build list with unique labels
label_list = []
for tag_str in train_data.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

            
# Display label list and length 
print(f'There are {len(train_data)} data samples, with {len(label_list)} possible classes.', '\n' 
      f'The Label list includes {label_list}')
# Split the tags column to get the unique labels
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in train_data['tags'].values])))


# Create label map
label_map = {l: i for i, l in enumerate(labels)}

print(f'label_map = {label_map},\n length = {len(label_map)}')
# Add onehot features for every label
train_tag_data = train_data.copy()
for label in label_list:
    train_tag_data[label] = train_tag_data['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)

# Display head
train_tag_data.head()
# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in train_tag_data['tags'].values]))
labels_set = set(labels_list)
print("There is {} unique labels including {}".format(len(labels_set), labels_set))


# Histogram of label instances
tag_labels = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tag_labels, y=tag_labels.index, orient='h')
# function for cooocurence matrix plotting
def make_cooccurence_matrix(labels):
    numeric_data = train_tag_data[labels]; 
    c_matrix = numeric_data.T.dot(numeric_data)
    sns.heatmap(c_matrix)
    return c_matrix
    
# Compute the co-ocurrence matrix
make_cooccurence_matrix(label_list)
# plot weather element cooccurence matrix
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']
make_cooccurence_matrix(weather_labels)
# plot land-use element classes cooccurence matrix
land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']
make_cooccurence_matrix(land_labels)
images = [train_data[train_data['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images, labels_set)):
    img = mpimg.imread('planet/planet/train-jpg' + '/' + image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))
# Determining if the length of the train and test dataset csv file equals the actual number of images in the folder

# Assign train and the two test dataset paths
# train path
train_img_dir = pathlib.Path('planet/planet/train-jpg')
train_img_path = sorted(list(train_img_dir.glob('*.jpg')))

# test path
test_img_dir = pathlib.Path('planet/planet/test-jpg')
test_img_path = sorted(list(test_img_dir.glob('*.jpg')))

# additional test path
test_add_img_dir = pathlib.Path('test-jpg-additional')
test_add_img_path = sorted(list(test_add_img_dir.glob('*/*.jpg')))

# Length Confirmation
assert len(train_img_path) == len(train_data)
print(len(test_img_path)+len(test_add_img_path))
# define input size
input_size = 64
# creating x_train and y_train
x_train = []
y_train = []

for f, tags in tqdm(train_data.values, miniters=1000):
    img = cv2.imread('planet/planet/train-jpg/{}.jpg'.format(f))
    img = cv2.resize(img, (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(img)
    y_train.append(targets)
        
x_train = np.array(x_train, np.float32)
y_train = np.array(y_train, np.uint8)

print(x_train.shape)
print(y_train.shape)
# creating x_test
x_test = []

test_jpg_dir = 'planet/planet/test-jpg'
test_image_names = os.listdir(test_jpg_dir)

n_test = len(test_image_names)
test_classes = test_data.iloc[:n_test, :]
add_classes = test_data.iloc[n_test:, :]

test_jpg_add_dir = 'test-jpg-additional/test-jpg-additional'
test_add_image_names = os.listdir(test_jpg_add_dir)

for img_name, _ in tqdm(test_classes.values, miniters=1000):
    img = cv2.imread(test_jpg_dir + '/{}.jpg'.format(img_name))
    x_test.append(cv2.resize(img, (64, 64)))
    
for img_name, _ in tqdm(add_classes.values, miniters=1000):
    img = cv2.imread(test_jpg_add_dir + '/{}.jpg'.format(img_name))
    x_test.append(cv2.resize(img, (64, 64)))

x_test = np.array(x_test, np.float32)
print(x_test.shape)
# split the train data into train and validation data sets
X_train = x_train[ :33000]
Y_train = y_train[ :33000]

X_valid = x_train[33000: ]
Y_valid = y_train[33000: ]
# specify sizes (batch and model input) and number of input channels
input_size = 64
input_channels = 3
batch_size = 64
# Add model parameters including dropout, layers and activation function
base_model = VGG16(include_top=False,
                   weights='imagenet',
                   input_shape=(input_size, input_size, input_channels))

model = Sequential()
model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))

model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(17, activation='sigmoid'))

path = '/kaggle/working'
os.chdir(path)   

# define model training optimizer parameters
optimizer  = SGD(lr=0.01)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
                ModelCheckpoint(filepath='weights/best_weights',
                                 save_best_only=True,
                                 save_weights_only=True)]
model.summary()
# train model
history = model.fit(x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
                  batch_size=batch_size,verbose=2, epochs=15,callbacks=callbacks,shuffle=True)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
p_valid = model.predict(X_valid, batch_size = batch_size, verbose=1)

print(fbeta_score(Y_valid, np.array(p_valid) > 0.18, beta=2, average='samples'))
y_pred = []
p_test = model.predict(x_test, batch_size=batch_size, verbose=2)
y_pred.append(p_test)
labels1 = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']
result = np.array(y_pred[0])
for i in range(1, len(y_pred)):
    result += np.array(y_pred[i])
result = pd.DataFrame(result, columns=labels)
# Translating the probability predictions to the unique labels
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.loc[[i]]
    a = a.apply(lambda x: x>0.2, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))
ls
# Replacing the tags columns with the predicted labels
test_data['tags'] = preds
test_data.head()
# Converting the dataframe to a csv file for submission
test_data.to_csv('amazon_sample_submission10.csv', index=False)