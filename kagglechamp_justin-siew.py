# Import necessities
import warnings as w
w.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import time
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.applications.xception import Xception
from keras import backend as K

from google.colab import files
# https://stackoverflow.com/questions/49088159/add-a-folder-with-20k-of-images-into-google-colaboratory
!pip install PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)
# Download and unzip data which was preprocessed and uploaded to drive previously
fileId = drive.CreateFile({'id': '1L9i_Ka1Tqg3hmB0T4CuxZWsy9eoLKZ6w'}) 
print (fileId['title']) # Datasets.zip
fileId.GetContentFile('Datasets.zip') 

!unzip Datasets.zip -d ./
# dimensions of our images.
img_width, img_height = 299, 299

original_data_dir = '/content/Datasets/Original Training/'
train_data_dir = '/content/Datasets/Training Images/'
validation_data_dir = '/content/Datasets/Validation/'
test_data_dir = '/content/Datasets/Test/'
epochs = 30
batch_size = 64
input_shape = (img_width, img_height, 3)
# Image Data Generator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_width, img_height),
        batch_size = 1,
        class_mode = 'categorical',
        shuffle = False)

# Create base pre-trained model
base_model = Xception(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalMaxPooling2D()(x)
x = Dropout(0.5)(x)
# Add fully-connected layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.7)(x)
# and softmax layer with 18 output nodes
predictions = Dense(18, activation='softmax')(x)

# train this
model = Model(inputs=base_model.input, outputs=predictions)
# first: train only the top layers (added by us)
# i.e. freeze all original Xception layers
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# Callbacks
best_model = '/content/Shopee_Xception_finetune.h5'
best_checkpoint = ModelCheckpoint(best_model, monitor='val_loss', verbose=1, save_best_only=True)
training_history = model.fit_generator(train_generator, 
                                       epochs = 10,
                                       validation = validation_generator,
                                       callbacks = [best_checkpoint])
model.save('/content/Shopee_Keras_Xception_top_trained_model.h5')
# load best checkpoint model for finetuning Xception weights
model = load_model(best_model)
# train only the middle and exit flow blocks as described in the Xception architecture
# keep low-level abstractions
# https://arxiv.org/abs/1610.02357
for layer in model.layers[:66]:
   layer.trainable = False
for layer in model.layers[66:]:
   layer.trainable = True
from keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])
training_history = model.fit_generator(train_generator, 
                                       epochs = 15,
                                       validation = validation_generator,
                                       callbacks = [best_checkpoint])


# load best checkpoint model for prediction
model = load_model(best_model)
# summarize history for accuracy
plt.plot(training_history.history['acc'])
plt.plot(training_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# Generate Predictions
test_files_names = test_generator.filenames
predictions = model.predict_generator(test_generator)
# For submission
predictions_sub = np.argmax(predictions, axis=1)
predictions_df = pd.DataFrame(predictions_sub, columns = ['category'])
predictions_df.insert(0, "id", test_files_names)
predictions_df['id'] = predictions_df['id'].map(lambda x: x.lstrip('Test/Test_').rstrip('.jpg'))
predictions_df['id'] = pd.to_numeric(predictions_df['id'], errors = 'coerce')
predictions_df.sort_values('id', inplace = True)
predictions_df.to_csv('/content/predictions_df.csv', index = False)

files.download('/content/predictions_df.csv')
# For ensembling
Xception_predictions_df = pd.DataFrame(predictions)
Xception_predictions_df.insert(0, "id", test_files_names)
Xception_predictions_df['id'] = Xception_predictions_df['id'].map(lambda x: x.lstrip('Test/Test_').rstrip('.jpg'))
Xception_predictions_df['id'] = pd.to_numeric(Xception_predictions_df['id'], errors = 'coerce')
Xception_predictions_df.sort_values('id', inplace = True)
Xception_predictions_df.to_csv('/content/Xception_predictions_df.csv', index = False)

files.download('/content/Xception_predictions_df.csv')
# download previously saved predictions
fileId = drive.CreateFile({'id': '1vAJ_NjMdlaRKT9GfZ8e3_X-jzLkscitr'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.80198).csv')  # Save Drive file as a local file

fileId = drive.CreateFile({'id': '1eyh5Ye1Q0Tl5EsToXrkuu3uaBCbo8Z3E'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.81150).csv')  # Save Drive file as a local file

fileId = drive.CreateFile({'id': '1M7gRwSUSO30LDuyovgeJPP7dQwZv-F5z'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.82474).csv')  # Save Drive file as a local file

fileId = drive.CreateFile({'id': '1g-OgQMBLHYCEevRbOKadj3ZoB6fe-ngx'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.82205).csv')  # Save Drive file as a local file

fileId = drive.CreateFile({'id': '1inZBPJIjBau3PzVuRwYbRAA88wDepUhj'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.82391).csv')  # Save Drive file as a local file

fileId = drive.CreateFile({'id': '1qPX855fVCJgawuj3zBPUwwPsw0uGbmcj'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.83550).csv')  # Save Drive file as a local file

fileId = drive.CreateFile({'id': '1qskR_uDFqxotHDSAiqsG5ZYLWksMdkE6'})
print (fileId['title']) 
fileId.GetContentFile('Xception_predictions_df(0.80715).csv')  # Save Drive file as a local file
df1 = pd.read_csv('/content/Xception_predictions_df(0.80198).csv')
df2 = pd.read_csv('/content/Xception_predictions_df(0.81150).csv')
df3 = pd.read_csv('/content/Xception_predictions_df(0.82474).csv')
df4 = pd.read_csv('/content/Xception_predictions_df(0.82205).csv')
df5 = pd.read_csv('/content/Xception_predictions_df(0.82391).csv')
df6 = pd.read_csv('/content/Xception_predictions_df(0.83550).csv')
df7 = pd.read_csv('/content/Xception_predictions_df(0.80715).csv')
# take a subset of previous predictions and ensemble them to submit for prediction. Here we just include all of them.
ensemble_df = df1.copy()
remaining_predictions = [df2, df3, df4, df5, df6, df7]

for df in remaining_predictions:
  ensemble_df = ensemble_df.append(df)
  
ensemble_df = ensemble_df.groupby('id', as_index=False).mean().reset_index(drop=True)
id_col = ensemble_df.id
ensemble_df = ensemble_df.drop('id', axis=1)
ensemble_preds = np.array(ensemble_df)
ensemble_preds = np.argmax(ensemble_preds, axis=1)
predictions_df = pd.DataFrame(ensemble_preds, columns = ['category'])
predictions_df = pd.concat([id_col.reset_index(drop=True), predictions_df], axis=1)
predictions_df.to_csv('/content/predictions_df.csv', index = False)

files.download('/content/predictions_df.csv')