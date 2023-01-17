import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow_hub as hub
import random


import os



labels = pd.read_csv('../input/dog-breed-identification/labels.csv')
labels.head()
labels["breed"].value_counts().plot.bar(figsize=(20, 10));
filenames = os.listdir('../input/dog-breed-identification/train')
filenames.sort()
filenames[:20]
df = pd.DataFrame({'filenames': filenames,
                   'breeds': labels['breed']})
df.head()
sample=random.choice(filenames)
image = load_img("../input/dog-breed-identification/train/"+sample)
plt.imshow(image)

train_df,val_df = train_test_split(df,test_size = 0.2,random_state = 42)
train_df.shape,val_df.shape
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_data = train_datagen.flow_from_dataframe( 
    train_df,
    "../input/dog-breed-identification/train/", 
    x_col='filenames',
    y_col='breeds',
    target_size=(224,224),
    class_mode='categorical',
    batch_size=32
)
valid_datagen = ImageDataGenerator(rescale = 1./255)
val_data = valid_datagen.flow_from_dataframe(val_df,
                                             "../input/dog-breed-identification/train/",
                                             x_col = 'filenames',
                                             y_col = 'breeds',
                                             target_size = (224,224),
                                             class_mode = 'categorical',
                                             batch_size = 32)

test_files = os.listdir("../input/dog-breed-identification/test/")
test_files.sort()
test_df = pd.DataFrame({
    'test_files': test_files
})
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_dataframe(
    test_df, 
    "../input/dog-breed-identification/test/", 
    x_col='test_files',
    y_col=None,
    class_mode=None,
    target_size=(224,224),
    batch_size=32,
    shuffle=False
)
INPUT_SHAPE = [None, 224, 224, 3] 

OUTPUT_SHAPE = 120 

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
    print("Building model with:", MODEL_URL)

  
    model = tf.keras.Sequential([
       hub.KerasLayer(MODEL_URL), 
       tf.keras.layers.Dense(units=OUTPUT_SHAPE, 
                          activation="softmax")])

  
    model.compile(
       loss=tf.keras.losses.CategoricalCrossentropy(), 
       optimizer=tf.keras.optimizers.Adam(), 
       metrics=["accuracy"])

 
    model.build(INPUT_SHAPE) 
  
    return model
model = create_model()
model.summary()
early_stopping = EarlyStopping(monitor="val_accuracy",
                                                  patience=3) 

def train_model():
    model = create_model()

    model.fit(x=train_data,
            epochs=50,
            validation_data=val_data,
            validation_freq=1, 
            callbacks=early_stopping)
    return model
  
    
model = train_model()
preds = model.predict(test_data,verbose = 1)
preds[:10]
sub = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')
sub.head()
labels = (train_data.class_indices)
labels = list(labels.keys())
df = pd.DataFrame(data=preds,
                 columns=labels)

columns = list(df)
columns.sort()
df = df.reindex(columns=columns)

filenames = sub["id"]
df["id"]  = filenames

cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.head(5)
df.to_csv('submission_df.csv',index = False)
