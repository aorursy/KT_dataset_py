import datetime as dt

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

sns.set_style('whitegrid')





import os

from keras.applications import xception

from keras.preprocessing import image

from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



from scipy.stats import uniform



from tqdm import tqdm

from glob import glob





from keras.models import Model, Sequential

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Masking

from keras.utils import np_utils, to_categorical





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

        

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

#copying the pretrained models to the cache directory

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)



#copy the Xception models

!cp ../input/keras-pretrained-models/xception* ~/.keras/models/

#show

!ls ~/.keras/models
base_folder = '../input/plant-seedlings-classification'

train_data_folder = os.path.join(base_folder, 'train')



#get the plant categories

categories = os.listdir(train_data_folder)

abbreviations = ['LSB', 'SP', 'FH', 'Ch', 'SB', 'Ma', 'CW', 'Cl', 'BG', 'SFC', 'SM', 'CC']

len_categories = len(categories)

#read the images from train folder

image_count = {}

train_data = []

for i, plant in tqdm(enumerate(categories)):

    plant_folder_name = os.path.join(train_data_folder, plant)

    plant_name = plant

    image_count[plant] = []

    for path in os.listdir(os.path.join(plant_folder_name)):

        image_count[plant].append(plant)

        train_data.append(['train/{}/{}'.format(plant, path), i, plant])

#create a dataframe

train_img = pd.DataFrame(train_data, columns=['filepath', 'id', 'class'])
#read the images from test folder



test_data = []



for file in os.listdir('../input/plant-seedlings-classification/test'):

    test_data.append(['test/{}'.format(file), file])

test_data = pd.DataFrame(test_data, columns=['filepath', 'filename'])



print('SHAPE OF TEST DATA: ', test_data.shape)

#show dataframe

test_data.head()
#show image count per class

for key,value in image_count.items():

    print("{0} -> {1} images".format(key, len(value)))
#show dataframe

train_img.head()
#show number of images

train_img.shape[0]
#get image samples per class and label them as traning data

sample_per_class = 200

train_data = pd.concat(train_img[train_img['class']==label][:sample_per_class] for label in categories ).sample(frac=1)

#rest the index

train_data.index = np.arange(train_data.shape[0])
# function to get an image

def read_img(filepath, size):

    img = image.load_img(os.path.join(base_folder, filepath), target_size=size)

    #convert image to array

    img = image.img_to_array(img)

    return img
#show sample images

nb_rows = 5

nb_cols = 5

fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 10))

plt.suptitle('SAMPLE IMAGES')

for i in range(0, nb_rows):

    for j in range(0, nb_cols):

        axs[i, j].xaxis.set_ticklabels([])

        axs[i, j].yaxis.set_ticklabels([])

        axs[i, j].imshow((read_img(train_data['filepath'][np.random.randint(100)], (224,224)))/255.)
INPUT_SIZE = 255

X_train  = np.zeros((len(train_data), INPUT_SIZE, INPUT_SIZE, train_data.shape[1]), dtype='float')



for i, file in tqdm(enumerate(train_data['filepath'])):

    img = read_img(file, (INPUT_SIZE, INPUT_SIZE))

    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))

    X_train[i] = x
print('Train Image Shape: ', X_train.shape)

print('Train Image Size: ', X_train.size)
y = to_categorical(train_data['id'])

train_x, train_val, y_train, y_val = train_test_split(X_train, y, test_size=0.2, random_state=101)
xception_b  = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

bf_train_x = xception_b.predict(train_x, batch_size=32, verbose=1)

bf_train_val = xception_b.predict(train_val, batch_size=32, verbose=1)
print('TRAIN DATA SHAPE: ', bf_train_x.shape)

print('TRAIN DATA SIZE: ', bf_train_x.size)

print('VALIDATION DATA SHAPE: ', bf_train_val.shape)

print('VALIDATION DATA SIZE: ', bf_train_val.size)
#keras Sequential model

model = Sequential()

model.add(Dense(units = 256 , activation = 'relu', input_dim=bf_train_x.shape[1]))

model.add(Dense(units = 64 , activation = 'relu'))

model.add(Dense(units = len_categories, activation = 'sigmoid'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

model.summary()
#train the model @ 100 epochs

history = model.fit(bf_train_x, y_train, epochs=200, batch_size=32)
fig, ax = plt.subplots(1,2,figsize=(14,5))

ax[0].set_title('TRAINING LOSS')

ax[1].set_title('TRAINING ACCURACY')





ax[0].plot(history.history['loss'], color= 'salmon',lw=2)

ax[1].plot(history.history['accuracy'], color= 'green',lw=2)
#predict the validation data

predictions = model.predict_classes(bf_train_val)

y_true = y_val.argmax(1)
#print classification report

print(classification_report(y_true, predictions))
#confusion matrix

con_mat = confusion_matrix(y_true, predictions)



plt.figure(figsize=(15,10))

plt.title('CONFUSION MATRIX', fontsize=20)



sns.heatmap(con_mat, cmap='coolwarm', yticklabels=abbreviations, xticklabels=abbreviations, annot=True)

plt.xlabel('True Class')

plt.ylabel('Predicted Class')



plt.savefig('Confusion Matrix.png', dpi=480)
X_test = np.zeros((len(test_data), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, filepath in tqdm(enumerate(test_data['filepath'])):

    img = read_img(filepath, (INPUT_SIZE, INPUT_SIZE))

    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))

    X_test[i] = x
bf_test = xception_b.predict(X_test, batch_size=32, verbose=1)

test_prediction = model.predict_classes(bf_test)
submission = pd.DataFrame(columns=['file', 'species'])

submission['file'] = test_data['filename']

submission['species'] = [categories[i] for i in test_prediction]



submission.to_csv('submission.csv', index=False)