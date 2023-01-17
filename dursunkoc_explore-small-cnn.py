import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.utils import to_categorical
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Input, Conv2D, multiply, LocallyConnected2D
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
classes = os.listdir("../input/train")
all_images_class = [os.listdir("../input/train/"+c) for c in classes]
int_to_classes = {i:classes[i] for i in range(len(classes))}
classes_to_int = {classes[i]:i for i in range(len(classes))}
df = pd.DataFrame({"n_images": [len(x) for x in all_images_class]}, index=classes)
df.index.name = "Specie"
df.plot(kind="bar", figsize=(15,10))
df.sort_values("n_images", ascending=False)
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
input_shape = (200,200,3)
np.random.seed(42)

def preprocessImagesToArray(img_name, specie):
    img = cv2.imread("../input/train/"+specie+"/"+img_name, cv2.IMREAD_COLOR)
    img = segment_plant(img)
    return classes_to_int[specie],cv2.resize(img_to_array(img), input_shape[:-1])

all_data = [preprocessImagesToArray(all_images_class[i][j], classes[i]) for i in range(len(classes)) for j in range(len(all_images_class[i]))]
labels = to_categorical(np.array(list(map(lambda tup: tup[0], all_data))),num_classes=len(classes))
features = np.array(list(map(lambda tup: tup[1], all_data)), dtype="float") / 255.0
columns = 4
rows = 5

fig, axs = plt.subplots(rows,columns, figsize=(20,20))

for i in range(rows*columns):
    axs[int(i/columns), i%columns].imshow(features[i])
(train_features, test_features, train_labels, test_labels) = train_test_split(features,labels,test_size=0.20, random_state=42, shuffle=True)
the_most_probable_guess = np.matlib.repmat(to_categorical(classes_to_int['Loose Silky-bent']), len(test_labels), 1)
baseline_pred = np.argmax(the_most_probable_guess, axis=1)
baseline_true = np.argmax(test_labels, axis=1)

f1_score(baseline_true, baseline_pred, average="micro")
def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score
EPOCHS = 20
INIT_LR = 1e-3
BS = 64

small_conv_nn = Sequential()

small_conv_nn.add(Conv2D(10, (5, 5), padding="same", input_shape=input_shape))
small_conv_nn.add(Activation("relu"))
small_conv_nn.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
small_conv_nn.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
small_conv_nn.add(Activation("relu"))
small_conv_nn.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
small_conv_nn.add(Conv2D(30, (5, 5), padding="same", input_shape=input_shape))
small_conv_nn.add(Activation("relu"))
small_conv_nn.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
small_conv_nn.add(Flatten())

small_conv_nn.add(Dense(2048))
small_conv_nn.add(Activation("relu"))
small_conv_nn.add(Dense(12))
small_conv_nn.add(Activation("softmax"))

small_conv_nn.compile(loss="categorical_crossentropy", optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS), metrics=[f1_score])
small_conv_nn.summary()
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
check_pt = ModelCheckpoint(
    'small_conv_nn_{epoch:02d}_{val_f1_score:.4f}.hdf5', 
    monitor='val_f1_score', 
    verbose=1, 
    save_best_only=False, 
    save_weights_only=False, 
    period=1
)


h_small_conv_nn = small_conv_nn.fit_generator(generator = aug.flow(train_features, train_labels, batch_size=BS), 
                            validation_data = (test_features, test_labels), 
                            steps_per_epoch = len(train_features) // BS, 
                            epochs = EPOCHS,
                            callbacks=[check_pt],
                            verbose = 1)
plt.plot(np.arange(0, EPOCHS), h_small_conv_nn.history["f1_score"], label="train_f1_score")
plt.plot(np.arange(0, EPOCHS), h_small_conv_nn.history["val_f1_score"], label="val_f1_score")
plt.title("Training Loss and Accuracy on  crop classification")
plt.xlabel("Epoch #")
plt.ylabel("Loss/F1-Score")
plt.legend(loc="lower right")
input_shape = (200,200,3)
np.random.seed(42)

def preprocessImagesToArray_pred(img_name):
    img = cv2.imread("../input/test/"+img_name, cv2.IMREAD_COLOR)
    img = segment_plant(img)
    return cv2.resize(img_to_array(img), input_shape[:-1])

all_test_images = os.listdir("../input/test/")
all_test_data = [preprocessImagesToArray_pred(img_name) for img_name in all_test_images]
features_test = np.array(all_test_data, dtype="float") / 255.0
predictions_test = small_conv_nn.predict_classes(features_test)
import pandas as pd
preds = np.column_stack((all_test_images,np.array(list(map(lambda i: int_to_classes[i], predictions_test)))))
submission = pd.DataFrame(data=preds, columns=["file","species"])
submission.to_csv('submission.csv', index=False)