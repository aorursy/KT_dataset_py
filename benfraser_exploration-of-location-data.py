!pip install reverse-geocode
import folium

import matplotlib.pyplot as plt

import numpy as np

import os

import pandas as pd

import pickle

import plotly.express as px

import reverse_geocode

import seaborn as sns

import shutil



from keras.callbacks import ModelCheckpoint

from keras.layers import MaxPooling2D, Conv2D, Dense, Flatten, Dropout, GlobalAveragePooling2D

from keras.models import Sequential, load_model

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image



from pathlib import Path

from PIL.ExifTags import TAGS, GPSTAGS

from PIL import Image



from skimage.feature import hog

from skimage.io import imread, imshow

from skimage.transform import resize

from skimage import exposure



from sklearn.model_selection import train_test_split
# if using gpu - confirm

import tensorflow as tf

tf.test.gpu_device_name()
np.random.seed(1)

tf.random.set_seed(1)
def extract_exif(filename):

    """ Extract img EXIF data """

    image = Image.open(filename)

    image.verify()

    return image._getexif()





def extract_exif_labelled(exif_data):

    """ Extract EXIF data with formatted labels """

    labelled_data = {}

    for (key, val) in exif_data.items():

        labelled_data[TAGS.get(key)] = val

    return labelled_data





def extract_geotags(exif_data):

    """ Obtain better formatted geotag data """

    if not exif_data:

        raise ValueError("EXIF metadata not found.")

    geotags = {}

    for (idx, geotag) in TAGS.items():

        if geotag == 'GPSInfo':

            if idx not in exif_data:

                raise ValueError("No EXIF geotagging found")

            for (key, val) in GPSTAGS.items():

                if key in exif_data[idx]:

                    geotags[val] = exif_data[idx][key]

    return geotags





def lat_long_alt_from_geotag(geotags):

    """ Obtain decimal lat, long and altitude from geotags """

    lat = dms_to_decimal(geotags['GPSLatitudeRef'], geotags['GPSLatitude'])

    long = dms_to_decimal(geotags['GPSLongitudeRef'], geotags['GPSLongitude'])

    

    # obtain altitude data and process, if it exists

    altitude = None

    try:

        alt = geotags['GPSAltitude']

        altitude = alt[0] / alt[1]

        

        # multiple by -1 if below sea level

        if geotags['GPSAltitudeRef'] == 1: 

            altitude *= -1

    except KeyError:

        altitude = 0

  

    return lat, long, altitude





def dms_to_decimal(lat_long_ref, deg_min_sec):

    """ Convert degrees, minutes, seconds tuples into decimal

        lat and lon values. Given to 5 decimal places - more 

        than sufficient for commercial GPS """

    

    degrees = deg_min_sec[0][0] / deg_min_sec[0][1]

    minutes = deg_min_sec[1][0] / deg_min_sec[1][1] / 60.0

    seconds = deg_min_sec[2][0] / deg_min_sec[2][1] / 3600.0

    

    if lat_long_ref in ['S', 'W']:

        degrees = -degrees

        minutes = -minutes

        seconds = -seconds

        

    return round(degrees + minutes + seconds, 5)
example_filename = '/kaggle/input/geolocated-imagery-dataset-scotland/300-399/322.jpg'

exif = extract_exif(example_filename)

labeled = extract_exif_labelled(exif)



for key, val in labeled.items():

    print(f"{key} : {val}")
geo_data = extract_geotags(exif)



for key, val in geo_data.items():

    print(f"{key} : {val}")
coords = lat_long_alt_from_geotag(geo_data)

print(coords)
img_name, img_path = [], []

latitudes, longitudes, altitudes = [], [], []

img_width, img_height = [], []

makes, models = [], []

time_dates = []



for dirname, _, filenames in os.walk('/kaggle/input'):

    

    for filename in filenames:

        

        if filename.endswith('.jpg'):

        

            file_path = os.path.join(dirname, filename)

        

            exif_data = extract_exif(file_path)

            exif_labels = extract_exif_labelled(exif_data)

            geo_data = extract_geotags(exif_data)

            lat, long, alt = lat_long_alt_from_geotag(geo_data)

        

            img_name.append(filename)

            img_path.append(file_path)

            latitudes.append(lat)

            longitudes.append(long)

            altitudes.append(alt)

            img_width.append(exif_labels.get('ImageWidth', 0))

            img_height.append(exif_labels.get('ImageLength', 0))

            makes.append(exif_labels.get('Make', 'Unknown'))

            models.append(exif_labels.get('Model', 'Unknown'))

            time_dates.append(exif_labels.get('DateTime', 0))
# reverse geocode our coordinates for the city

coord_pairs = [(lat,long) for lat, long in zip(latitudes, longitudes)]

cities = [x['city'] for x in reverse_geocode.search(coord_pairs)]
metadata_df = pd.DataFrame({ 'filename' : img_name, 'filepath' : img_path, 

                             'img_width' : img_width, 'img_height' : img_height, 

                             'make' :makes, 'model' : models,

                             'latitude' : latitudes, 'longitude' : longitudes, 

                             'altitude' : altitudes, 'time_date' : time_dates,

                             'city' : cities})



metadata_df.head()
metadata_df['img_height'].value_counts()
plt.figure(figsize=(10,5))

metadata_df['altitude'].plot()

plt.ylabel("Altitude (m)")

plt.xlabel("Image number")

plt.show()
date_index_df = metadata_df.copy()

date_index_df['Date'] = pd.to_datetime(date_index_df['time_date'], format='%Y:%m:%d %H:%M:%S')

date_index_df.sort_values(by=['Date'], inplace=True, ascending=True)

date_index_df.reset_index(inplace=True, drop=True)



plt.figure(figsize=(10,5))

date_index_df['altitude'].plot()

plt.ylabel("Altitude (m)")

plt.xlabel("Image number")

plt.show()
plt.figure(figsize=(12,6))

values = metadata_df['city'].value_counts()

sns.barplot(x = values.index.values, y = values.values)

plt.xticks(rotation=90)

plt.show()
# obtain towns / cities with top ten image counts

values = metadata_df['city'].value_counts()[:10]



plt.figure(figsize=(10,5))

sns.barplot(x = values.index.values, y = values.values)

plt.xticks(rotation=90)

plt.show()



top_towns = list(values.index.values)

top_towns
fig = plt.figure(figsize=(12, 6))



for i, example in enumerate(top_towns):

    

    ax = fig.add_subplot(2, 5, i+1)

    ax.set_xticks([])

    ax.set_yticks([])

    

    class_imgs = metadata_df[metadata_df['city'] == example]

    example_img_path = class_imgs.iloc[0]['filepath']

    

    example_img = imread(example_img_path)

    

    ax.imshow(example_img)

    

    ax.set_xlabel(example)
def plot_data_coords(row):

    folium.Circle(location=[row.latitude, row.longitude],

                  color='crimson',

                  tooltip = "<h5 style='text-align:center;font-weight: bold'>Img Name : "+row.filename+"</h5>"+

                            "<hr style='margin:10px;'>"+

                            "<ul style='color: #444;list-style-type:circle;align-item:left;"+

                            "padding-left:20px;padding-right:20px'>"+

                            "<li>Town : "+str(row.city)+"</li>"+

                            "<li>Lat : "+str(row.latitude)+"</li>"+

                            "<li>Long : "+str(row.longitude)+"</li>"+

                            "<li>Altitude : "+str(row.altitude)+"</li>"+

                            "<li>Time date : "+str(row.time_date)+"</li></ul>",

                  radius=20, weight=6).add_to(m)



    

m = folium.Map(location=[metadata_df['latitude'].mean(), 

                         metadata_df['longitude'].mean()], 

               tiles='OpenStreetMap',

               min_zoom=7, max_zoom=12, zoom_start=7.5)





# iterate through all rows and plot coords

metadata_df.apply(plot_data_coords, axis = 1)



m
colors = ['red', 'blue', 'gray', 'darkred', 'black', 'orange', 'beige', 'green', 

          'purple', 'lightgreen', 'darkblue', 'lightblue', 'darkgreen', 'darkpurple',

          'lightred', 'cadetblue', 'lightgray', 'pink']



# dict comp to form unique color for each town

town_colors = { town : color for town, color in zip(top_towns, colors[:len(top_towns)]) }



# select only data containing our selected towns / cities

top_towns_df = metadata_df[metadata_df['city'].isin(top_towns)]
def plot_top_towns(row):

    

    marker_colour = town_colors[row['city']]

    

    folium.Circle(location=[row.latitude, row.longitude],

                  color=marker_colour,

                  tooltip = "<h5 style='text-align:center;font-weight: bold'>Img Name : "+row.filename+"</h5>"+

                            "<hr style='margin:10px;'>"+

                            "<ul style='color: #444;list-style-type:circle;align-item:left;"+

                            "padding-left:20px;padding-right:20px'>"+

                            "<li>Town : "+str(row.city)+"</li>"+

                            "<li>Lat : "+str(row.latitude)+"</li>"+

                            "<li>Long : "+str(row.longitude)+"</li>"+

                            "<li>Altitude : "+str(row.altitude)+"</li>"+

                            "<li>Time date : "+str(row.time_date)+"</li></ul>",

                  radius=20, weight=6).add_to(m)

    

m = folium.Map(location=[top_towns_df['latitude'].mean(), 

                         top_towns_df['longitude'].mean()], 

               tiles='OpenStreetMap',

               min_zoom=7, max_zoom=12, zoom_start=7.5)



# iterate through all rows and plot coords

top_towns_df.apply(plot_top_towns, axis = 1)



m
# obtain our data classes (output labels) using top towns from the data

classes = [town.lower() for town in top_towns]



# create our new directories - pathlib Path to avoid preexisting errors

base_dir = os.path.join(os.getcwd(), 'Base_Data')

Path(base_dir).mkdir(parents=True, exist_ok=True)

train_dir = os.path.join(base_dir, 'train')

validation_dir = os.path.join(base_dir, 'validation')

test_dir = os.path.join(base_dir, 'test')



# form each of the dirs above

for directory in [train_dir, validation_dir, test_dir]:

    Path(directory).mkdir(parents=True, exist_ok=True)

    

# create sub-directories for each town class for train, val and test dirs

for town_class in classes:

    # create sub-directories within training directory

    current_train_dir = os.path.join(train_dir, town_class)

    Path(current_train_dir).mkdir(parents=True, exist_ok=True)

    

    # repeat for validation dir

    current_val_dir = os.path.join(validation_dir, town_class)

    Path(current_val_dir).mkdir(parents=True, exist_ok=True)

    

    # repeat for test dir

    current_test_dir = os.path.join(test_dir, town_class)

    Path(current_test_dir).mkdir(parents=True, exist_ok=True)
# create training, validation and test splits for all images using the file paths

X_path = top_towns_df['filepath'].values

y = top_towns_df['city'].values



# first split - training + validation split combined, and seperate 10% test split.

X_train_val_paths, X_test_paths, y_train, y_test = train_test_split(X_path, y, 

                                                                    test_size=0.1, 

                                                                    random_state=1, 

                                                                    stratify=y)



# second split - 75% training and 25% validation data

X_train_paths, X_val_paths, y_train, y_val = train_test_split(X_train_val_paths, 

                                                              y_train, 

                                                              test_size=0.25, 

                                                              random_state=1, 

                                                              stratify=y_train)
# get counts of class labels within each split

trg_towns, trg_counts =  np.unique(y_train, return_counts=True)

val_towns, val_counts =  np.unique(y_val, return_counts=True)

test_towns, test_counts =  np.unique(y_test, return_counts=True)



# plot number of classes within each data split for confirmation

fig = plt.figure(figsize=(12, 4))

split_types = ['Training', 'Validation', 'Test']



for i, data_split in enumerate([trg_counts, val_counts, test_counts]):

    

    ax = fig.add_subplot(1, 3, i+1)

    sns.barplot(x = trg_towns, y = data_split)

    plt.xticks(rotation=90)

    plt.title(split_types[i])



plt.show()
!ls /kaggle/working/Base_Data/train/
%%time



# copy training data

for i, img_location in enumerate(X_train_paths):

    class_label = y_train[i].lower()

    img_name = f"train_{i}.jpg"

    src_loc = img_location

    dest_loc = os.path.join(train_dir, class_label, img_name)

    

    # resize img and then move

    img = Image.open(src_loc)

    img_new = img.resize((504,378), Image.ANTIALIAS)

    img_new.save(dest_loc, 'JPEG', quality=90)

    

    # move img without resizing using shutil

    #_ = shutil.copyfile(src_loc, dest_loc)



# copy validation data

for i, img_location in enumerate(X_val_paths):

    class_label = y_val[i].lower()

    img_name = f"validation_{i}.jpg"

    src_loc = img_location

    dest_loc = os.path.join(validation_dir, class_label, img_name)

    

    # resize img and then move

    img = Image.open(src_loc)

    img_new = img.resize((504,378), Image.ANTIALIAS)

    img_new.save(dest_loc, 'JPEG', quality=90)

    

    #_ = shutil.copyfile(src_loc, dest_loc)

    

# copy test data

for i, img_location in enumerate(X_test_paths):

    class_label = y_test[i].lower()

    img_name = f"test_{i}.jpg"

    src_loc = img_location

    dest_loc = os.path.join(test_dir, class_label, img_name)

    

    # resize img and then move

    img = Image.open(src_loc)

    img_new = img.resize((504,378), Image.ANTIALIAS)

    img_new.save(dest_loc, 'JPEG', quality=90)

    

    #_ = shutil.copyfile(src_loc, dest_loc)
img_height, img_width = 299, 299

batch_size = 10



# training data augmentation - rotate, shear, zoom and flip

train_datagen = ImageDataGenerator(

    rotation_range = 30,

    rescale = 1.0 / 255.0,

    shear_range = 0.2,

    zoom_range = 0.2,

    horizontal_flip = True,

    vertical_flip=True)



# no augmentation for test data - only rescale

test_datagen = ImageDataGenerator(rescale = 1. / 255.0)



# generate batches of augmented data from training data

train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')



# generate val data from val dir

validation_generator = test_datagen.flow_from_directory(

    validation_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    class_mode='categorical')



nb_train_samples = len(train_generator.classes)

nb_validation_samples = len(validation_generator.classes)



# create pandas dataframes for our train data

training_data = pd.DataFrame(train_generator.classes, columns=['classes'])

testing_data = pd.DataFrame(validation_generator.classes, columns=['classes'])
def create_CNN():

    """ Basic CNN with 4 Conv layers, each followed by a max pooling """

    cnn_model = Sequential()

    

    # four Conv layers with max pooling

    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))

    cnn_model.add(MaxPooling2D(2, 2))

    

    # flatten output and feed to dense layer, via dropout layer

    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.5))

    cnn_model.add(Dense(512, activation='relu'))

    

    # add output layer - softmax with 10 outputs

    cnn_model.add(Dense(10, activation='softmax'))

    

    cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    

    return cnn_model
CNN_model = create_CNN()

CNN_model.summary()
#history = CNN_model.fit_generator(train_generator, epochs=30, 

#                                  validation_data=validation_generator, shuffle=True)
# save model as a HDF5 file with weights + architecture

#CNN_model.save('Basic_CNN_model_1.h5')



# save the history of training to a datafile for later retrieval

#with open('train_history_basic_CNN_model_1.pickle', 'wb') as pickle_file:

#        pickle.dump(history.history, pickle_file)
# if already trained - import history file and training weights

CNN_model = load_model('/kaggle/input/basic-cnn-model/Basic_CNN_model_1.h5')



# get history of trained model

with open('/kaggle/input/basic-cnn-model/train_history_basic_CNN_model_1.pickle', 'rb') as handle:

    history = pickle.load(handle)
hist_dict_1 = history



trg_loss = hist_dict_1['loss']

val_loss = hist_dict_1['val_loss']



trg_acc = hist_dict_1['accuracy']

val_acc = hist_dict_1['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



# plot losses and accuracies for training and validation 

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(1, 2, 1)

plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

ax.set_ylabel("Loss")

ax.set_xlabel("Epochs")

plt.legend(loc='best')



ax = fig.add_subplot(1, 2, 2)

plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Accuracy")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.show()
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(299, 299), 

                                                  batch_size=4, class_mode='categorical')



test_loss, test_accuracy = CNN_model.evaluate_generator(test_generator)

print(f"Test accuracy: {test_accuracy}")
from keras.applications import xception
# create our pretrained convolutonal base from xception

conv_base = xception.Xception(weights='imagenet', include_top=False)
conv_base.summary()
for layer in conv_base.layers:

  layer.trainable = False
tl_xception = Sequential()



# add pre-trained xception base

tl_xception.add(conv_base)



# flatten and add dense layer, with dropout

tl_xception.add(GlobalAveragePooling2D())

tl_xception.add(Dropout(0.5))

tl_xception.add(Dense(256, activation='relu'))



# output softmax, with 10 classes

tl_xception.add(Dense(10, activation='softmax'))



tl_xception.compile(loss='categorical_crossentropy', 

                    optimizer='adam', 

                    metrics=['accuracy'])
# set up a check point for our model - save only the best val performance

save_path ="tl_xception_1_best_weights.hdf5"



trg_checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', 

                                 verbose=1, save_best_only=True, mode='max')



trg_callbacks = [trg_checkpoint]
# batch steps before an epoch is considered complete (trg_size / batch_size):

steps_per_epoch = np.ceil(nb_train_samples/batch_size)



# validation batch steps (val_size / batch_size):

val_steps_per_epoch = np.ceil(nb_validation_samples/batch_size)
#history = tl_xception.fit(train_generator, epochs=25, 

#                          steps_per_epoch=steps_per_epoch, 

#                          validation_data=validation_generator, 

#                          validation_steps=val_steps_per_epoch,

#                          callbacks=trg_callbacks,

#                          shuffle=True)
# save model as a HDF5 file with weights + architecture

#tl_xception.save('tl_xception_1.h5')



# save the history of training to a datafile for later retrieval

#with open('tl_xception_history_1.pickle', 

#          'wb') as pickle_file:

#        pickle.dump(history.history, pickle_file)



loaded_model = False
# if already trained - import history file and training weights

tl_xception = load_model('/kaggle/input/inception-transfer-learning-model/tl_xception_1_model.hdf5')



# get history of trained model

with open('/kaggle/input/inception-transfer-learning-model/tl_xception_history_1.pickle', 'rb') as handle:

    history = pickle.load(handle)

    

loaded_model = True
# if loaded model set history accordingly

if loaded_model:

    hist_dict_2 = history

else:

    hist_dict_2 = history.history



trg_loss = hist_dict_2['loss']

val_loss = hist_dict_2['val_loss']



trg_acc = hist_dict_2['accuracy']

val_acc = hist_dict_2['val_accuracy']



epochs = range(1, len(trg_acc) + 1)



# plot losses and accuracies for training and validation 

fig = plt.figure(figsize=(14,6))

ax = fig.add_subplot(1, 2, 1)

plt.plot(epochs, trg_loss, marker='o', label='Training Loss')

plt.plot(epochs, val_loss, marker='x', label='Validation Loss')

plt.title("Training / Validation Loss")

ax.set_ylabel("Loss")

ax.set_xlabel("Epochs")

plt.legend(loc='best')



ax = fig.add_subplot(1, 2, 2)

plt.plot(epochs, trg_acc, marker='o', label='Training Accuracy')

plt.plot(epochs, val_acc, marker='^', label='Validation Accuracy')

plt.title("Training / Validation Accuracy")

ax.set_ylabel("Accuracy")

ax.set_xlabel("Epochs")

plt.legend(loc='best')

plt.show()
test_generator = test_datagen.flow_from_directory(test_dir, 

                                                  target_size=(299, 299), 

                                                  batch_size=5, 

                                                  class_mode='categorical')



test_loss, test_accuracy = tl_xception.evaluate(test_generator, steps=10)

print(f"Test accuracy: {test_accuracy}")
# get class labels dict containing index of each class for decoding predictions

class_labels = train_generator.class_indices



# obtain a reverse dict to convert index into class labels

reverse_class_index = {i : class_label for class_label, i in class_labels.items()}
def process_and_predict_img(image_path, model):

    """ Utility function for making predictions for an image. """

    img_path = image_path

    img = image.load_img(img_path, target_size=(299, 299))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = test_datagen.standardize(x)

    predictions = model.predict(x)

    return img, predictions
def top_n_predictions(predict_probs, top_n_labels=3):

    """ Obtain top n prediction indices for array of predictions """

    top_indices = np.argpartition(predict_probs[0], -top_n_labels)[-top_n_labels:]

    

    # negate prediction array to sort in descending order

    sorted_top = top_indices[np.argsort(-predict_probs[0][top_indices])]

    

    # dict comp to create dict of labels and probs

    labels = {"label_" + str(i + 1) : (reverse_class_index[index].capitalize(), 

                                       predict_probs[0][index]) for i, index in enumerate(sorted_top)}

    return labels
img, prediction = process_and_predict_img(test_dir + '/caol/test_26.jpg', 

                                          model=tl_xception)

top_labels = top_n_predictions(prediction, 

                               top_n_labels=3)

plt.imshow(img)

plt.title(f"Location: {top_labels['label_1'][0]}")

plt.show()



print("Top predictions:")

for label in top_labels:

    print("- {0}: {1:.2f}%".format(top_labels[label][0], top_labels[label][1] * 100))
example_test_i = np.random.permutation(len(y_test))[:12]

example_test_img = X_test_paths[example_test_i]

example_test_y = y_test[example_test_i]



# create fig to display 12 different predictions

fig = plt.figure(figsize=(15,9))

img_num = 0



for i in range(12):

    ax = fig.add_subplot(3, 4, img_num + 1)

    

    img_path = example_test_img[img_num]

    

    # make prediction on image - select desired model (e.g. CNN_basic, or tl_xception)

    img, predictions = process_and_predict_img(img_path, model=tl_xception)

    top_labels = top_n_predictions(predictions, top_n_labels=3)

    

    prediction_string = ""

    for label in top_labels:

        prediction_string += f"- {top_labels[label][0]}: {top_labels[label][1]*100:.2f}% \n"

    

    ax.imshow(img)

    

    #title = reverse_class_index[np.argmax(predictions,axis=-1)[0]].capitalize()

    ax.set_title(f"Town: {example_test_y[img_num]}")

    ax.set_xlabel(f"Predictions: \n{prediction_string}")

    ax.set_xticks([])

    ax.set_yticks([])

    

    img_num += 1



plt.tight_layout()

plt.show()
# create fig to display 12 different predictions

fig = plt.figure(figsize=(15,9))

img_num = 0



for i in range(12):

    ax = fig.add_subplot(3, 4, img_num + 1)

    

    img_path = example_test_img[img_num]

    

    # make prediction on image - select desired model (e.g. CNN_basic, or tl_xception)

    img, predictions = process_and_predict_img(img_path, model=CNN_model)

    top_labels = top_n_predictions(predictions, top_n_labels=3)

    

    prediction_string = ""

    for label in top_labels:

        prediction_string += f"- {top_labels[label][0]}: {top_labels[label][1]*100:.2f}% \n"

    

    ax.imshow(img)

    

    #title = reverse_class_index[np.argmax(predictions,axis=-1)[0]].capitalize()

    ax.set_title(f"Town: {example_test_y[img_num]}")

    ax.set_xlabel(f"Predictions: \n{prediction_string}")

    ax.set_xticks([])

    ax.set_yticks([])

    

    img_num += 1



plt.tight_layout()

plt.show()
try:

    shutil.rmtree(base_dir)

except OSError as e:

    print("Error: %s : %s" % (base_dir, e.strerror))