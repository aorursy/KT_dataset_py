# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import datetime, os

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
#loading dataframes

train_df = pd.read_csv('/kaggle/input/rsna-bone-age/boneage-training-dataset.csv')

test_df = pd.read_csv('/kaggle/input/rsna-bone-age/boneage-test-dataset.csv')



#appending file extension to id column for both training and testing dataframes

train_df['id'] = train_df['id'].apply(lambda x: str(x)+'.png')

test_df['Case ID'] = test_df['Case ID'].apply(lambda x: str(x)+'.png') 



train_df.head()
#finding out the number of male and female children in the dataset

#creating a new column called gender to keep the gender of the child as a string

train_df['gender'] = train_df['male'].apply(lambda x: 'male' if x else 'female')

print(train_df['gender'].value_counts())

sns.countplot(x = train_df['gender'])
#oldest child in the dataset

print('MAX age: ' + str(train_df['boneage'].max()) + ' months')



#youngest child in the dataset

print('MIN age: ' + str(train_df['boneage'].min()) + ' months')



#mean age is

mean_bone_age = train_df['boneage'].mean()

print('mean: ' + str(mean_bone_age))



#median bone age

print('median: ' +str(train_df['boneage'].median()))



#standard deviation of boneage

std_bone_age = train_df['boneage'].std()



#models perform better when features are normalised to have zero mean and unity standard deviation

#using z score for the training

train_df['bone_age_z'] = (train_df['boneage'] - mean_bone_age)/(std_bone_age)



print(train_df.head())
#plotting a histogram for bone ages

train_df['boneage'].hist(color = 'green')

plt.xlabel('Age in months')

plt.ylabel('Number of children')

plt.title('Number of children in each age group')
train_df['bone_age_z'].hist(color = 'violet')

plt.xlabel('bone age z score')

plt.ylabel('Number of children')

plt.title('Relationship between number of children and bone age z score')
#Relationship between age and gender with a categorical scatter plot (swarmplot)

sns.swarmplot(x = train_df['gender'], y = train_df['boneage'])
#distribution of age within each gender 

male = train_df[train_df['gender'] == 'male']

female = train_df[train_df['gender'] == 'female']

fig, ax = plt.subplots(2,1)

ax[0].hist(male['boneage'], color = 'blue')

ax[0].set_ylabel('Number of boys')

ax[1].hist(female['boneage'], color = 'red')

ax[1].set_xlabel('Age in months')

ax[1].set_ylabel('Number of girls')

fig.set_size_inches((10,7))
#splitting train dataframe into traininng and validation dataframes

df_train, df_valid = train_test_split(train_df, test_size = 0.2, random_state = 0)
import matplotlib.image as mpimg

for filename, boneage, gender in train_df[['id','boneage','gender']].sample(4).values:

    img = mpimg.imread('/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/'+ filename)

    plt.imshow(img)

    plt.title('Image name:{}  Bone age: {} years  Gender: {}'.format(filename, boneage/12, gender))

    plt.axis('off')

    plt.show()
#library required for image preprocessing

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from  keras.applications.xception import preprocess_input 



#reducing down the size of the image 

img_size = 256



train_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

val_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)



#train data generator

train_generator = train_data_generator.flow_from_dataframe(

    dataframe = df_train,

    directory = '/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset',

    x_col= 'id',

    y_col= 'bone_age_z',

    batch_size = 32,

    seed = 42,

    shuffle = True,

    class_mode= 'other',

    flip_vertical = True,

    color_mode = 'rgb',

    target_size = (img_size, img_size))



#validation data generator

val_generator = val_data_generator.flow_from_dataframe(

    dataframe = df_valid,

    directory = '/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset',

    x_col = 'id',

    y_col = 'bone_age_z',

    batch_size = 32,

    seed = 42,

    shuffle = True,

    class_mode = 'other',

    flip_vertical = True,

    color_mode = 'rgb',

    target_size = (img_size, img_size))



#test data generator

test_data_generator = ImageDataGenerator(preprocessing_function = preprocess_input)



test_generator = test_data_generator.flow_from_directory(

    directory = '/kaggle/input/rsna-bone-age/boneage-test-dataset',

    shuffle = True,

    class_mode = None,

    color_mode = 'rgb',

    target_size = (img_size,img_size))
test_X, test_Y = next(val_data_generator.flow_from_dataframe( 

                            df_valid, 

                            directory = '/kaggle/input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset',

                            x_col = 'id',

                            y_col = 'bone_age_z', 

                            target_size = (img_size, img_size),

                            batch_size = 2523,

                            class_mode = 'other'

                            )) 
def plot_it(history):

    '''function to plot training and validation error'''

    fig, ax = plt.subplots( figsize=(20,10))

    ax.plot(history.history['mae_in_months'])

    ax.plot(history.history['val_mae_in_months'])

    plt.title('Model Error')

    plt.ylabel('error')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Val'], loc='upper right')

    ax.grid(color='black')

    plt.show()
from keras.metrics import mean_absolute_error

def mae_in_months(x_p, y_p):

    '''function to return mae in months'''

    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense,Flatten

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

from tensorflow.keras import Sequential



model_1 = tf.keras.applications.xception.Xception(input_shape = (img_size, img_size, 3),

                                           include_top = False,

                                           weights = 'imagenet')

model_1.trainable = True

model_2 = Sequential()

model_2.add(model_1)

model_2.add(GlobalMaxPooling2D())

model_2.add(Flatten())

model_2.add(Dense(10, activation = 'relu'))

model_2.add(Dense(1, activation = 'linear'))





#compile model

model_2.compile(loss ='mse', optimizer= 'adam', metrics = [mae_in_months] )



#model summary

model_2.summary()
# Load the TensorBoard notebook extension

%load_ext tensorboard

logs_dir = '.\logs'

%tensorboard --logdir {logs_dir}
#early stopping

early_stopping = EarlyStopping(monitor='val_loss',

                              min_delta=0,

                              patience= 5,

                              verbose=0, mode='auto')



#model checkpoint

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)



#tensorboard callback

logdir = os.path.join(logs_dir,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback =  TensorBoard(logdir, histogram_freq = 1)



#reduce lr on plateau

red_lr_plat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)



callbacks = [tensorboard_callback,early_stopping,mc, red_lr_plat]





#fit model

history = model_2.fit_generator(train_generator,

                            steps_per_epoch = 315,

                            validation_data = val_generator,

                            validation_steps = 1,

                            epochs = 50,

                            callbacks= callbacks)

history

%tensorboard --logdir logs

plot_it(history)

model_2.load_weights('best_model.h5')

pred = mean_bone_age + std_bone_age*(model_2.predict(test_X, batch_size = 32, verbose = True))

test_months = mean_bone_age + std_bone_age*(test_Y)



ord_ind = np.argsort(test_Y)

ord_ind = ord_ind[np.linspace(0, len(ord_ind)-1, 8).astype(int)] # take 8 evenly spaced ones

fig, axs = plt.subplots(4, 2, figsize = (15, 30))

for (ind, ax) in zip(ord_ind, axs.flatten()):

    ax.imshow(test_X[ind, :,:,0], cmap = 'bone')

    ax.set_title('Age: %fY\nPredicted Age: %fY' % (test_months[ind]/12.0, 

                                                           pred[ind]/12.0))

    ax.axis('off')

fig.savefig('trained_image_predictions.png', dpi = 300)

fig, ax = plt.subplots(figsize = (7,7))

ax.plot(test_months, pred, 'r.', label = 'predictions')

ax.plot(test_months, test_months, 'b-', label = 'actual')

ax.legend(loc = 'upper right')

ax.set_xlabel('Actual Age (Months)')

ax.set_ylabel('Predicted Age (Months)')
test_generator.reset()

y_pred = model_2.predict_generator(test_generator)

predicted = y_pred.flatten()

predicted_months = mean_bone_age + std_bone_age*(predicted)

filenames=test_generator.filenames

results=pd.DataFrame({"Filename":filenames,

                      "Predictions": predicted_months})

results.to_csv("results.csv",index=False)