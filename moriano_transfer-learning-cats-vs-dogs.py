# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
import pandas as pd
import random
train_data = []  # This will later be split in validation too
test_data = []
for file in listdir("../input/train"):
    some_number = random.randint(1,100)
    label = "1" if "dog" in file else "0" 
    if len(test_data) >= 1000 or some_number < 85:
        train_data.append([file, label])
    else:
        test_data.append([file, label])
        
train = pd.DataFrame(train_data, columns=["filename", "class"])
test = pd.DataFrame(test_data, columns = ["filename", "class"])
train.head(10)
test.head(10)
print("Train size", len(train))
print("Test size", len(test))

for label in ["0", "1"]:
    print("------------")
    print("\tTrain has", len(train[train["class"]==label]), label)
    print("\tTest has", len(test[test["class"]==label]), label)
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE=32
train_image_generator = ImageDataGenerator(rescale=1./255, 
                                           rotation_range=90, 
                                           horizontal_flip=True, 
                                           vertical_flip=True,
                                           validation_split=0.15)
train_generator = train_image_generator.flow_from_dataframe(train, "../input/train", seed=42,
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="binary",
                                                    subset="training",
                                                    shuffle=True,      
                                                    save_format="jpeg")

validation_generator = train_image_generator.flow_from_dataframe(train, "../input/train", seed=42,
                                                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode="binary",
                                                    subset="validation",
                                                    shuffle=False,                  
                                                    save_format="jpeg")
from keras.applications import vgg16
model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), pooling="max")
for layer in model.layers[:-5]:
        layer.trainable = False
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.models import Model, Sequential

# Although this part can be done also with the functional API, I found that for this simple models, this becomes more intuitive
transfer_model_vgg16 = Sequential()
for layer in model.layers:
    transfer_model_vgg16.add(layer)
transfer_model_vgg16.add(Dense(512, activation="relu"))  # Very important to use relu as activation function, search for "vanishing gradiends" :)
transfer_model_vgg16.add(Dense(1, activation="sigmoid")) # Finally our activation layer! we use 2 outputs as we have either cats or dogs
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(transfer_model_vgg16).create(prog='dot', format='svg'))
from keras import optimizers
adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)

transfer_model_vgg16.compile(adam, 
                       loss="binary_crossentropy",
                      metrics=["accuracy"])
vgg16_model_history = transfer_model_vgg16.fit_generator(train_generator, 
                                             steps_per_epoch = train_generator.n // BATCH_SIZE,
                                             validation_data = validation_generator,
                                             validation_steps = validation_generator.n // BATCH_SIZE,
                                            epochs=7)
from IPython.display import Image, display
def plot_prediction(image_path, label):
    display(Image(filename=image_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT))
    prediction = "dog"
    confidence = label
    if label < 0.5:
        prediction = "cat"
        confidence = (1-label)
    legend = "The image %s above is a %s with a confidence of %.2f%% %f" % (image_path, prediction, confidence*100, label)
    print(legend)
import cv2
from skimage import io

def build_batches(df, has_labels=True, limit=500, batch_size=BATCH_SIZE, produce="images"):
    """
    produce: Can be either "images" in which case an array of normalized images is returned or 
             "paths" in which case, a string with the full dir is returned
    """
    X = []
    y = []
    paths = []
    i = 0
    for _, row in df.iterrows():
        if has_labels:
            y.append(row["class"])
        raw_image_path = "../input/train/" if has_labels else "../input/test/"
        raw_image_path += row["filename"]
        raw_image = io.imread(raw_image_path)
        raw_image = cv2.resize(raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        X.append(raw_image)
        paths.append(raw_image_path)
        i += 1
        if i == limit:
            break
        if i > 0 and i % batch_size == 0:
            X = np.array(X)
            y = np.array(y)
            X = X / 255
            
            if produce == "images":
                yield X, y
            else:
                yield paths, y
            paths = []
            X = []
            y = []

    X = np.array(X)
    y = np.array(y)
    
    X = X / 255
    
    if produce == "images":
        yield X, y         
    else:
        yield paths, y
samples = 1000
transfer_model_vgg16.evaluate_generator(build_batches(test, limit=samples), steps=samples/BATCH_SIZE, verbose=True)
some_predictions = transfer_model_vgg16.predict_generator(build_batches(test, limit=12, batch_size=1), steps=12, verbose=True)
idx = 0
for mini_batch_files, mini_batch_labels in build_batches(test, limit=samples, batch_size=1, produce="paths"):
    mini_batch_file = mini_batch_files[0]
    mini_batch_label = mini_batch_labels[0]
    predicted_label = some_predictions[idx][0]
    idx += 1
    #print(mini_batch_file, mini_batch_label, predicted_label)
    plot_prediction(mini_batch_file, predicted_label)
samples = 1000
some_predictions = transfer_model_vgg16.predict_generator(build_batches(test, limit=samples, batch_size=1), steps=samples, verbose=True)

print("Total predictions", some_predictions.shape)
idx = 0
errors = 0
for mini_batch_files, mini_batch_labels in build_batches(test, limit=samples, batch_size=1, produce="paths"):
    mini_batch_file = mini_batch_files[0]
    mini_batch_label = mini_batch_labels[0]
    predicted_label = some_predictions[idx][0]
    if abs(float(mini_batch_label) - float(predicted_label)) > 0.5:
        errors += 1
        if errors < 10:
            plot_prediction(mini_batch_file, predicted_label)
    idx += 1
print("Total errors...", errors)
my_limit = 12500
i = 0
output_df = []
for file in listdir("../input/test/"):    
    output_df.append([file, file.split(".")[0]])
    i += 1
    if i == my_limit:
        break
output = pd.DataFrame(output_df, columns=["filename", "id"])
print(len(output))
output.head()
results = transfer_model_vgg16.predict_generator(build_batches(output, limit=-1, has_labels=False, batch_size=64), steps=12500/64, verbose=True)
results.shape
output["label"] = results

output.head(15)
stop = 10
for idx, row in output.iterrows():
    path = "../input/test/" + row["id"] + ".jpg"
    plot_prediction(path, row["label"])
    stop -= 1
    if stop == 0:
        break
del output["filename"]
output.head(10)
len(output)
output.to_csv("submission_file.csv", index=False)