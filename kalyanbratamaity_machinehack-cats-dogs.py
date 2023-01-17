# import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

from sklearn.model_selection import train_test_split

import os
print("Cats & Dogs Dataset Folder Contains:", os.listdir("../input/Data"))
# read the train images
TRAIN_IMAGE_FOLDER_PATH = "../input/Data/Train/"

full_paths, targets = list(), list()

for folder in os.listdir(TRAIN_IMAGE_FOLDER_PATH):
    folder_path = os.path.join(TRAIN_IMAGE_FOLDER_PATH, folder)
   
    for file in os.listdir(folder_path):
        new_file_name = folder + "." + file
        image_path = os.path.join(folder_path, file)
        
        full_paths.append(image_path)
        targets.append(folder.lower()[:-1])
        
        
dataset = pd.DataFrame()
dataset['filename'] = full_paths
dataset['category'] = targets

dataset = dataset.sample(frac=1).reset_index(drop=True)

dataset.head(10)
target_counts = dataset['category'].value_counts()
print("Number of dogs in the dataset:{}".format(target_counts['dog']))
print("Number of cats in the dataset:{}".format(target_counts['cat']))

def get_side(img, side_type, side_size=5):
    height, width, channel = img.shape
    if side_type == "horizontal":
        return np.ones((height, side_size, channel), dtype = np.float32) * 255
    
    return np.ones((side_size, width, channel), dtype = np.float32) * 255



def show_gallery(show="both"):
    n, counter = 100, 0
    images, vertical_images = [], []
    
    rng_state = np.random.get_state()
    np.random.shuffle(full_paths)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

    for path, target in zip(full_paths, targets):
        if target != show and show != "both":
            continue
        counter += 1
        if counter % 100 == 0:
            break
            
        # load image as .jpg file
        img = load_img(path, target_size=(WIDTH, HEIGHT))
        # converting .jpg file to numpy array
        img = img_to_array(img)
        
        
        hside = get_side(img, side_type="horizontal")
        images.append(img)
        images.append(hside)
        
        
        if counter %10 == 0:
            himage = np.hstack((images))
            vside = get_side(himage, side_type="vertical")
            vertical_images.append(himage)
            vertical_images.append(vside)
            
            images = []

            
    gallery = np.vstack((vertical_images))
    plt.figure(figsize=(12,12))
    plt.xticks([])
    plt.yticks([])
    title = {"both" : "Dogs and Cats",
             "cat" : "Cats",
             "dog" : "Dogs"}
    plt.title("100 samples of {} of the dataset".format(title[show]))
    plt.imshow(gallery.astype(np.uint8))
            
WIDTH, HEIGHT = 150, 150

show_gallery(show="cat")
show_gallery(show="dog")
show_gallery(show="both")
# plot diagnostic learning curves
def show_model_history(model_history, model_name):
    history = pd.DataFrame()
    history["Train Loss"] = model_history.history["loss"]
    history["Validation Loss"] = model_history.history["val_loss"]
    history["Train Accuracy"] = model_history.history["accuracy"]
    history["Train Loss"] = model_history.history["val_accuracy"]
    
    
    history.plot(figsize=(12,8))
    plt.title(" Convulutional Model {} Train and Validation Loss and Accuracy History".format(model_name))
    plt.show()
    
# define cnn model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", 
                        input_shape=(WIDTH, HEIGHT, 3)))
model.add(layers.Conv2D(32, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128, (3,3), activation="relu"))
model.add(layers.Conv2D(128, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.Conv2D(64, (3,3), activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation="sigmoid"))

# print the model
model.summary()

# compile the model
model.compile(loss="binary_crossentropy",
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=["accuracy"])

print("[INFO]: model compiled...")
# split the data to train & test
dataset_train, dataset_test = train_test_split(dataset,
                                                   test_size=0.2,
                                                   random_state=42)
# create the train data generator
train_datagen = ImageDataGenerator(rotation_range=15,
                                  rescale=1./255,
                                  shear_range=0.1,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  width_shift_range=0.1,
                                  height_shift_range=0.1)

train_datagenerator = train_datagen.flow_from_dataframe(dataframe=dataset_train,
                                                        x_col="image_path",
                                                        y_col="target",
                                                        target_size=(WIDTH, HEIGHT),
                                                        class_mode="binary",
                                                        batch_size=150)
# create the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_datagenerator = test_datagen.flow_from_dataframe(dataframe=dataset_test,
                                                        x_col="image_path",
                                                        y_col="target",
                                                        target_size=(WIDTH, HEIGHT),
                                                        class_mode="binary",
                                                        batch_size=150)
model_history = model.fit_generator(train_datagenerator,
                                   epochs=50, 
                                   validation_data=test_datagenerator,
                                   validation_steps=dataset_test.shape[0]//150,
                                   steps_per_epoch=dataset_train.shape[0]//150)
print("Train Accuracy:{:.3f}".format(model_history.history["accuracy"][-1]))
print("Test Accuracy:{:.3f}".format(model_history.history["val_accuracy"][-1]))
show_model_history(model_history=model_history, model_name="")
# save the model
model.save("model1_catsVSdogs_10epoch.h5")

# test data preparation
TEST_IMAGE_FOLDER_PATH = "../input/Data/Test/"
#print(os.listdir(TEST_IMAGE_FOLDER_PATH))
full_paths = list()
for file in os.listdir(TEST_IMAGE_FOLDER_PATH):
    full_path = os.path.join(TEST_IMAGE_FOLDER_PATH, file)
    full_paths.append(full_path)
    
test_data = pd.DataFrame({'filename' : full_paths})
test_data.shape
test_datagen = ImageDataGenerator(rescale=1/255.)
test_generator = test_datagen.flow_from_dataframe(test_data,target_size=(150,150),
                                                 shuffle=False,
                                                 class_mode=None,
                                                 batch_size=1,
                                                 seed=7)
predict = model.predict_generator(test_generator, steps=np.ceil(test_data.shape[0]/1))
predict = predict.tolist()
predict_class = []
for i in predict:
    if i[0] > 0.5:
        predict_class.append(1)
    else:
        predict_class.append(0)
test_data['Category'] = predict_class
test_data['File_Name'] = test_data.filename.str.split("/", expand=True)[4]
test_data = test_data.drop('filename', axis=1)
test_data = test_data[['File_Name', 'Category']]
test_data
test_data.to_csv('predicted.csv', index=False)