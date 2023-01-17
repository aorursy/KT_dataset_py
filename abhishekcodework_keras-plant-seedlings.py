import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import cv2
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.inception_v3 import preprocess_input as inception_preprocess
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
NUM_CATEGORIES = len(CATEGORIES)
SEED = 1
data_dir = '../input/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))
train = []
for category_id, category in enumerate(CATEGORIES):
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append(['train/{}/{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])
train.head(2)
train.shape
test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test = pd.DataFrame(test, columns=['filepath', 'file'])
test.head(2)
test.shape
fig = plt.figure(1, figsize=(NUM_CATEGORIES, NUM_CATEGORIES))
grid = ImageGrid(fig, 111, nrows_ncols=(NUM_CATEGORIES, NUM_CATEGORIES), axes_pad=0.05)
i = 0
for category_id, category in enumerate(CATEGORIES):
    for filepath in train[train['category'] == category]['file'].values[:NUM_CATEGORIES]:
        ax = grid[i]
        img = Image.open("../input/"+filepath)
        img = img.resize((240,240))
        ax.imshow(img)
        ax.axis('off')
        if i % NUM_CATEGORIES == NUM_CATEGORIES - 1:
            ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')
        i += 1
plt.show();
w=[]
v=[]
for category in CATEGORIES:
    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))
    w.append(category)
    v.append(len(os.listdir(os.path.join(train_dir, category))))
plt.figure(figsize=(25, 7))
plt.xlabel('Classes')  
plt.ylabel('Count')  
plt.title("Data Distribution for each class")
plt.bar(w,v)
plt.show()
def create_segmented_image(img):
    blurr = cv2.GaussianBlur(img,(5,5),0)
    hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)
    
    sensitivity = 30
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])
    
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    masked = mask > 0
    preprocessed = np.zeros_like(img,np.uint8)
    preprocessed[masked] = img[masked]

    return np.asarray(preprocessed)

from keras.layers import BatchNormalization
model1 = applications.ResNet50(weights = "imagenet", include_top=False, input_shape = (224, 224,3))
model2 = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (224, 224, 3))
model3 = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (299, 299, 3))
model1.summary()
x = model1.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x=BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x=BatchNormalization()(x)
predictions = Dense(12, activation="softmax")(x) 
x = model1.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x=BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x=BatchNormalization()(x)
predictions = Dense(12, activation="softmax")(x) 
model_final = Model(input = model1.input, output = predictions)
#compling our model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.01), metrics=["accuracy"])
for layer in model_final.layers[:-9]:
    layer.trainable = False

x = model3.output
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
x=BatchNormalization()(x)
predictions = Dense(12, activation="softmax")(x) 

model_final.fit_generator(
                    train_generator,
                    validation_data=val_generator,
                    epochs = 10,
                    shuffle= True,
                    callbacks = [early])
model_final = Model(input = model3.input, output = predictions)
#compling our model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.01), metrics=["accuracy"])
model_final = Model(input = model1.input, output = predictions)
#compling our model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.01), metrics=["accuracy"])
model_final.summary() #Model summary
gen = ImageDataGenerator(preprocessing_function = resnet_preprocess,
           # rescale=1./255, #only change
            width_shift_range=0.1,
            rotation_range=30.,
            validation_split=0.25,
            horizontal_flip=True,
            vertical_flip=True)
train_data_dir = "../input/train"
train_generator = gen.flow_from_directory(
                        train_data_dir,
                        target_size = (299, 299),
                        batch_size = 64, 
                        subset='training',
                        class_mode = "categorical")
val_generator = gen.flow_from_directory(train_data_dir,target_size = (299, 299),
                        batch_size = 64, 
                        subset='validation',
                        class_mode = "categorical")
checkpoint = ModelCheckpoint("inceptionv3.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=1, mode='auto')
for layer in model_final.layers[:-9]:
    layer.trainable = False
    #v3
for layer in model_final.layers[:-11]:
    layer.trainable = False
    #resnet
for l in model_final.layers:
    print(l.name, l.trainable)
model_final.fit_generator(
                    train_generator,
                    validation_data=val_generator,
                    epochs = 10,
                    shuffle= True,
                    callbacks = [early])
classes = train_generator.class_indices  
print(classes)
#Invert Mapping
classes = {v: k for k, v in classes.items()}
print(classes)
prediction = []
for filepath in test['filepath']:
    img = cv2.imread(os.path.join(data_dir,filepath))
    img = cv2.resize(img,(240,240))
    img = np.asarray(img)
    img = img.reshape(1,240,240,3)
    pred = model_final.predict(img)
    prediction.append(classes.get(pred.argmax(axis=-1)[0])) #Invert Mapping helps to map Label
test = test.drop(columns =['filepath']) #Remove file path from test DF
sample_submission.head()
pred = pd.DataFrame({'species': prediction})
test =test.join(pred)
test.to_csv('submission.csv', index=False)
test.head()