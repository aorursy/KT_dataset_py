import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px

from keras.models import Sequential 
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

consolidated_path = '../input/100-bird-species/consolidated/'
test_path = '../input/100-bird-species/test/'
train_path = '../input/100-bird-species/train/'
valid_path = '../input/100-bird-species/valid/'

def tuple_count(file_path, dataset):
    bird_count = []
    for file in os.listdir(file_path):
        bird_count.append((file, len(os.listdir(file_path + file)), dataset))
    return bird_count

consolidated = tuple_count(test_path, 'test') + tuple_count(valid_path, 'valid') + tuple_count(train_path, 'train')
count_df = pd.DataFrame.from_records(consolidated, columns =['Name', 'Count', 'From']) 

fig = px.bar(count_df, x='Name', y='Count', color='From')
fig.update_xaxes(visible=False)
fig.show()
def display_random_grid(ncols=5, ds_path=consolidated_path):
    fig, ax = plt.subplots(ncols=ncols, nrows=ncols, figsize=(15, 15))
    
    for i in range(ncols):
        for j in range(ncols):
            bird_species = random.choice(os.listdir(ds_path))
            random_bird_path = random.choice(os.listdir(ds_path + bird_species))
            random_bird = mpimg.imread(ds_path + bird_species + '/' + random_bird_path)
            ax[i, j].imshow(random_bird)
            ax[i, j].set_title(bird_species)
            ax[i, j].axis('off')
            
display_random_grid()
# cheching out the baseline model 

def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid', input_shape=input_shape))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='valid'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model 

num_classes = len(os.listdir(consolidated_path))
img_shape = (224, 224, 3)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_path,
        batch_size=300,
        target_size=(224, 224),
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        valid_path,
        batch_size=20,
        target_size=(224, 224),
        class_mode='categorical')

model = create_model(img_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50, 
      verbose=1)
import plotly.graph_objects as go

acc = history.history['acc']
xs = list(range(len(acc)))
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


fig = go.Figure()
fig.add_trace(go.Scatter(x=xs, y=acc, mode='lines+markers', name='Accuracy'))
fig.add_trace(go.Scatter(x=xs, y=val_acc, mode='lines+markers', name='Validation accuracy'))
fig.add_trace(go.Scatter(x=xs, y=loss, mode='lines+markers', name='Loss'))
fig.add_trace(go.Scatter(x=xs, y=val_loss, mode='lines+markers', name='Validation loss'))
fig.show()
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_path, 
                                                  target_size=(224, 224),
                                                  batch_size=1)
model.evaluate_generator(test_generator)
