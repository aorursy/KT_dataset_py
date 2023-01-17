import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing import image
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import os, os.path
import shutil
import numpy as np
from PIL import Image
# Opening data zip files
root = "../input/dogs-vs-cats-redux-kernels-edition/"
sample_submission = pd.read_csv(root + "sample_submission.csv")

with zipfile.ZipFile(root + 'test.zip',"r") as test:
    test.extractall(".")
    
with zipfile.ZipFile(root + 'train.zip',"r") as train:
    train.extractall(".")
        
# Creating paths to training and test data files
data_dir = '../working/train/'
test_dir_old = '../working/test/'

# Lists of image file names in training and test files
data_list = os.listdir(data_dir)
test_list = os.listdir(test_dir_old)

# Partitioning a section of training data for validation
train_valid_split = 0.9
train_number = int(train_valid_split * len(data_list))

# Creating new subdirectories for classes (cats and dogs)
os.mkdir('../working/train/cats/')
os.mkdir('../working/train/dogs/')
os.mkdir('../working/valid/')
os.mkdir('../working/valid/cats/')
os.mkdir('../working/valid/dogs/')
os.mkdir('../working/test/test/')

# Creating paths to new subdirectories
valid_dir = '../working/valid/'
test_dir_new = '../working/test/test/'
train_cats = '../working/train/cats/'
train_dogs = '../working/train/dogs/'
valid_cats = '../working/valid/cats/'
valid_dogs = '../working/valid/dogs/'
# Moving data into subdirectories

# Moving image files into training files for cats and dogs
for filename in data_list[:train_number]:
    category = filename.split('.')[0]
    if category == 'dog':
        shutil.move(data_dir+filename, train_dogs+filename)
    if category == 'cat':
        shutil.move(data_dir+filename, train_cats+filename)

# Moving image files into validation files for cats and dogs      
for filename in data_list[train_number:]:
    category = filename.split('.')[0]
    if category == 'dog':
        shutil.move(data_dir+filename, valid_dogs+filename)
    if category == 'cat':
        shutil.move(data_dir+filename, valid_cats+filename)

# Moving train data into train subdirectory 
for filename in test_list:
    shutil.move(test_dir_old+filename, test_dir_new+filename)
        
# Verifying directories contents
len_train_cats, len_train_dogs, len_valid_cats, len_valid_dogs, len_test = len(os.listdir(train_cats)), len(os.listdir(train_dogs)), len(os.listdir(valid_cats)), len(os.listdir(valid_dogs)), len(os.listdir(test_dir_new))
len_train = len_train_cats + len_train_dogs
len_valid = len_valid_cats + len_valid_dogs
print("There are {} cats and {} dogs in the training set. There are {} cats and {} dogs in the validation set. There are {} images in the test set.".format(len_train_cats, len_train_dogs, len_valid_cats, len_valid_dogs, len_test))
# Creating the data to plot
data_sets = ['Training', 'Validation', 'Testing']
cats = [len_train_cats, len_valid_cats, 0]
dogs = [len_train_dogs, len_valid_dogs, 0]
unlabelled = [0, 0, len_test]
labels = ['Cat Images', 'Dog Images', 'Unlabelled Images']

# Set position and width of bars
pos = list(range(len(data_sets)))
width = 0.25 
    
# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))

# Create a bar with cat data in position pos
plt.bar(pos, cats, width, alpha=1, color='#EE3224', label=labels[0]) 

# Create a bar with dogs data in position pos + some width buffer,
plt.bar([p + width for p in pos], dogs, width, alpha=1, color='#F78F1E', label=labels[1]) 

# Create a bar with unlabelled data,
# in position pos + some width buffer,
plt.bar([p + width*2 for p in pos], unlabelled, width, alpha=1, color='#FFC222', label=labels[2]) 

# Set the y axis label
ax.set_ylabel('Quantity in set', fontsize=14)

# Set the chart's title
ax.set_title('Categories of data in each set', fontsize=16)

# Set the position of the x ticks
ax.set_xticks([p + width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(data_sets, fontsize=14)

# Remove grid lines
ax.grid(None)

# Adding the legend and showing the plot
plt.legend(labels, loc='upper center')
plt.grid()
plt.show()                       
# Plotting images of only dogs

dimensions = 5

fig, axs = plt.subplots(dimensions, dimensions, figsize=(dimensions * 4, dimensions * 4))
for i in range(0, dimensions * dimensions):
    
    doggy_id = os.listdir(train_dogs)[i]
    
    row = i // dimensions
    col = i % dimensions
    img_PIL = Image.open(train_dogs+doggy_id)
    im_resized = img_PIL.resize((150,150))
    ax = axs[row][col]
    ax.imshow(im_resized)
    ax.set_title("Dog")
# Adding augmentations with datagenerator

train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_gen = ImageDataGenerator(
    rescale=1./255)

test_gen = ImageDataGenerator(
    rescale=1./255)

# Creating generators
train_generator = train_gen.flow_from_directory(data_dir, batch_size=64, class_mode='binary', target_size=(150, 150))
valid_generator = valid_gen.flow_from_directory(valid_dir, batch_size=64, class_mode='binary', target_size=(150, 150))
test_generator = test_gen.flow_from_directory(test_dir_old, batch_size=64, class_mode='binary', shuffle=False, target_size=(150, 150))
# Pretrained model

# Loading InceptionV3 pretrained on imagenet
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights='imagenet')
    
# Removing the last layers of the model
last_layer = 'mixed7'
model = Model(inputs=pre_trained_model.input, outputs=pre_trained_model.get_layer(last_layer).output)
model.summary()

# Creating a new model by adding different layers to the end of the cropped pre-trained model
new_model = tf.keras.models.Sequential([
    model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Setting the pretrained model to non-trainable
new_model.layers[0].trainable = False
new_model.summary()
# Add a learning rate scheduler
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Compiling the model
new_model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Training
NUM_EPOCHS = 3 
history = new_model.fit(train_generator, steps_per_epoch=len_train/64, epochs=NUM_EPOCHS, validation_data=(valid_generator), validation_steps=len_valid/64, callbacks=[learning_rate_reduction], verbose=1)
# Metrics

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting accuracy
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
plt.xlabel('epoch', fontsize=12)

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Binary Cross Entropy', fontsize=12)
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch', fontsize=12)
plt.show()

# Predicting the test image classes
predictions = new_model.predict(test_generator)
# Converting the output predictions to the submission format for scoring
predictions_list = predictions.tolist()
for index, item in enumerate(predictions_list):
    predictions_list[index] = float(item[0])
    
results_df = pd.DataFrame(
    {
        'id': pd.Series(test_generator.filenames), 
        'label': pd.Series(predictions_list)
    })
results_df['id'] = results_df.id.str.extract('(\d+)')
results_df['id'] = pd.to_numeric(results_df['id'], errors = 'coerce')
results_df.sort_values(by='id', inplace = True)

results_df.to_csv('submission.csv', index=False)
results_df.head(20)
# Plotting the images and labelling with class predicitions
dimensions = 5
fig, axs = plt.subplots(dimensions, dimensions, figsize=(dimensions * 4, dimensions * 4))
for i in range(0, dimensions * dimensions):
    
    doggy_id = i + 1 +100
    
    row = i // dimensions
    col = i % dimensions
    img_PIL = Image.open(test_dir_new+ f"{doggy_id}.jpg")
    im_resized = img_PIL.resize((150,150))
    ax = axs[row][col]
    ax.imshow(im_resized)
    ax.set_title(f"Prediction: {results_df.label[results_df.id == doggy_id].values[0]:.4f} ID: {doggy_id}")
ax = results_df['label'].plot.hist(bins=10, alpha=1)
sorted_list = results_df
sorted_list['uncertainty'] = abs(sorted_list['label']-0.5)
sorted_list = sorted_list.sort_values(by = ['uncertainty'])
sorted_list = sorted_list.reset_index(drop=True)
sorted_list.head(10)
dimensions = 6
from PIL import Image
fig, axs = plt.subplots(dimensions, dimensions, figsize=(dimensions * 4, dimensions * 4))
for i in range(0, dimensions * dimensions):
    
    doggy_id = sorted_list.id[i]
    
    row = i // dimensions
    col = i % dimensions
    img_PIL = Image.open(test_dir_new+ f"{doggy_id}.jpg")
    im_resized = img_PIL.resize((150,150))
    ax = axs[row][col]
    ax.imshow(im_resized)
    ax.set_title(f"Prediction: {sorted_list.label[sorted_list.id == doggy_id].values[0]:.4f} ID: {doggy_id}")
