import numpy as np

import cv2

import matplotlib.pyplot as plt

import time

weights_path = '../input/yolo-coco-data/yolov3.weights'

configuration_path = '../input/yolo-coco-data/yolov3.cfg'



probability_minimum = 0.5
network = cv2.dnn.readNetFromDarknet(configuration_path, weights_path)



layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

image_input = cv2.imread('../input/the-oxfordiiit-pet-dataset/images/images/Birman_136.jpg')



image_input_shape = image_input.shape



print(image_input_shape) 

%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

plt.show()

blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)



print(blob.shape)  # (1, 3, 416, 416)



blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)

print(blob_to_show.shape)



%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 5.0)

plt.imshow(blob_to_show)

plt.show()
network.setInput(blob)

output_from_network = network.forward(layers_names_output)

bounding_boxes = []

confidences = []

class_numbers = []

h = image_input_shape[0]

w = image_input_shape[1]
for result in output_from_network:

    for detection in result:

        scores = detection[5:]

        class_current = np.argmax(scores)



        confidence_current = scores[class_current]



        if confidence_current > probability_minimum:



            box_current = detection[0:4] * np.array([w, h, w, h])



            x_center, y_center, box_width, box_height = box_current.astype('int')

            x_min = int(x_center - (box_width / 2))

            y_min = int(y_center - (box_height / 2))



            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])

            confidences.append(float(confidence_current))


for i in range(len(bounding_boxes)):

    x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]

    box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]



    colour_box_current = (0, 0, 255)



    cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height),

                  colour_box_current, 2)

    text_box_current = '{:.3f}'.format(confidences[i])

    cv2.putText(image_input, text_box_current, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX,

                1.5, colour_box_current, 2)
%matplotlib inline

plt.rcParams['figure.figsize'] = (10.0, 10.0)

plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))

plt.show()

from pathlib import Path

import pandas as pd
# " ".join(str(i).split('_')[:-1])



path = Path("../input/the-oxfordiiit-pet-dataset/images/images/")

data = []

for i in path.iterdir():

    label = "_".join(str(i).split('/')[-1].split('_')[:-1])

    data.append((i, label))

data = pd.DataFrame(data, columns=['filename', 'label'])

data.head()
def crop_detected(filename):   

    input_image = cv2.imread(filename)

    if input_image is None:

        return (0, 0, 0, 0)

    input_image_shape = input_image.shape



    blob = cv2.dnn.blobFromImage(input_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)



    network.setInput(blob)

    output_from_network = network.forward(layers_names_output)



    bounding_box = []

    confidence = []

    h = input_image_shape[0]

    w = input_image_shape[1]

    





    for detection in output_from_network[0]:

        scores = detection[5:]

        class_current = np.argmax(scores)



        confidence_current = scores[class_current]



        if confidence_current > probability_minimum:



            box_current = detection[0:4] * np.array([w, h, w, h])



            x_center, y_center, box_width, box_height = box_current.astype('int')

            x_min = int(x_center - (box_width / 2))

            y_min = int(y_center - (box_height / 2))



            bounding_box = [x_min, y_min, int(box_width), int(box_height)]

            confidence = float(confidence_current)



    if len(bounding_box) > 0:

        x_min, y_min = bounding_box[0], bounding_box[1]

        box_width, box_height = bounding_box[2], bounding_box[3]



        return (x_min, y_min, box_width, box_height)

    else:

        return (0, 0, 0, 0)

    
import os

BASE = Path('/kaggle/working/cropped_cats')

if not os.path.exists(BASE):

    os.makedirs(BASE)

counter = 0

n = len(data)

for index, item in data.iterrows():

    filename = str(item['filename'])

    x, y, w, h = crop_detected(filename)

    if w == 0:

        continue

    if not os.path.exists(os.path.join(BASE, item['label'])):

        os.makedirs(os.path.join(BASE, item['label']))

    label_dir = os.path.join(BASE, item['label'])

    image = cv2.imread(filename)[y:y+h,x:x+w]

    cv2.imwrite(os.path.join(label_dir, filename.split('/')[-1]), image)

    

    counter +=1

    print("{}/{}".format(counter, n), end = "\r")
output = input_image[y_min:y_min+box_height,x_min:x_min+box_width]

plt.imshow(output)
from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.optimizers import RMSprop

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense
path = Path('../input/yolo-v3/cropped_cats')

train_data = [(str(img), str(label).split('/')[-1]) for label in path.iterdir() for img in label.iterdir()]

train_data = pd.DataFrame(train_data, columns=['filename', 'label'])

train, test = train_test_split(train_data, test_size=0.2)

n_labels = train_data['label'].nunique()
n_labels
# ../input/the-oxfordiiit-pet-dataset/images/images/Abyssinian_103.jpg

img = cv2.imread('../input/the-oxfordiiit-pet-dataset/images/images/staffordshire_bull_terrier_155.jpg')

plt.imshow(img)
IMAGE_SIZE = 128

IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE

BATCH_SIZE = 32



train_datagen = ImageDataGenerator(

    rescale=1./255,

    zoom_range=0.1,

    shear_range=0.1,

    horizontal_flip=True

    )



val_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_dataframe(dataframe = train, x_col = "filename",

                                                    y_col = "label", target_size = (IMAGE_SIZE, 

                                                                                    IMAGE_SIZE),

                                                    batch_size = BATCH_SIZE, 

                                                    class_mode = 'categorical')





val_generator = val_datagen.flow_from_dataframe(dataframe = test, x_col = "filename",

                                                 y_col = "label", target_size = (IMAGE_SIZE, 

                                                                                 IMAGE_SIZE),

                                                 batch_size = BATCH_SIZE, 

                                                 class_mode = 'categorical', 

                                                 shuffle = False)







input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

model = Sequential()



model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(n_labels))

model.add(Activation('sigmoid'))

    

model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=0.0001),

              metrics=['accuracy'])



steps_train = np.ceil(train.shape[0]/BATCH_SIZE)

steps_val = np.ceil(test.shape[0]/BATCH_SIZE)



def my_gen(gen):

    while True:

        try:

            data, labels = next(gen)

            yield data, labels

        except:

            print('fuck')

            pass

    



model.fit_generator(my_gen(train_generator), steps_per_epoch = steps_train, validation_data = None, epochs = 5, validation_steps = steps_val, verbose = 1)
model.evaluate_generator(my_gen(val_generator), steps_val)