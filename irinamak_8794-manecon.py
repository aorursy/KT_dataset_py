#import libraries 

import os

import numpy as np
import pandas as pd
import glob

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from matplotlib.image import imread
import matplotlib.pyplot as plt
%matplotlib inline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

from keras.preprocessing.image import image
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
font = {
    'family': 'serif',
    'color':  'darkred',
    'weight': 'bold',
    'size': 22
}
SEED = 257

TRAIN_DIR = '../input/hotdogs-spbu/train/train'
TEST_DIR = '../input/hotdogs-spbu/test/test'
categories = ['hot dog', 'not hot dog']
X, y = [], []

for category in categories:
    category_dir = os.path.join(TRAIN_DIR, category)
    for image_path in os.listdir(category_dir):
        X.append(imread(os.path.join(category_dir, image_path)))
        y.append(category)
#checking the correctness of the import
print (y[0])
plt.axis("off");
plt.imshow(X[0]);
#transforming labels into 1/0
y = [1 if x == 'hot dog' else 0 for x in y]
#train/test split
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=SEED)
# checking data shape: № of multicolored images 100x100
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# converting data to fit ANN
num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
# just in case (making sure channels are the same)
img_rows, img_cols = 100, 100
if K.image_data_format() == 'channels_first':
    x_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    x_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    x_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
#batch_size = 256
#epochs = 12 & 100 
#img_rows, img_cols = 100, 100
#num_classes = 2

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))

#model.compile( loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
#model.fit( x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Final structure    
#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax'))
# creating generator for image change
train_datagen = ImageDataGenerator(zoom_range=0.5, rotation_range=50,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                   horizontal_flip=True, fill_mode='nearest')
# let's see what happens with our pictures
img_id = 0
hotdogs = train_datagen.flow(x_train[img_id:img_id+1], y_train[img_id:img_id+1], batch_size=1)
hdog = [next(hotdogs) for i in range(0,5)]
fig, ax = plt.subplots(1,5, figsize=(16, 6))
l = [ax[i].imshow(hdog[i][0][0]) for i in range(0,5)]
# creating generator for image change
train_generator = train_datagen.flow(x_train, y_train, batch_size=30)
# settings for CNN
batch_size = 128
epochs = 10 # actual solution = 450
img_rows, img_cols = 100, 100
# Model structure
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=['accuracy']
)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=200,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test)
)
#check performance via graph
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,11))
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 11, 1))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 11, 1))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
# for 10 epochs (please, find empirical findings on ROC below)
roc_auc_score(y_test, model.predict(X_test))
#preparing fo submission
leaderboard_X = []
leaderboard_filenames = []

for image_path in os.listdir(TEST_DIR):
    leaderboard_X.append(imread(os.path.join(TEST_DIR, image_path)))
    leaderboard_filenames.append(image_path)
#predicting on leaderboard data
leaderboard_X = np.asarray(leaderboard_X)
predictions = model.predict(leaderboard_X)
#checking predictions on random sample
print(predictions[:,1][120])
# checking on graph
idx = 132

plt.axis("off");
if predictions[:,1][idx] > 0.5:
    plt.text(20, -5, 'HOT DOG!!!', fontdict=font)
else:
    plt.text(15, -5,'not hot dog...', fontdict=font)
plt.imshow(leaderboard_X[idx]);
# creating dataframe
submission = pd.DataFrame(
    {
        'image_id': leaderboard_filenames, 
        'image_hot_dog_probability': predictions[:,1]
    }
)
# checking submissions
submission.head()
# submission for kaggle
submission.to_csv('submitim200.csv', index=False)
#image generator = the same as used above +

#resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#for layer in resnet_model.layers:
    #if isinstance(layer, BatchNormalization):
        #layer.trainable = True
    #else:
        #layer.trainable = False
#model = Sequential()
#model.add(UpSampling2D())
#model.add(resnet_model.layers[0])
#model.add(GlobalAveragePooling2D())
#model.add(Dense(256, activation='relu'))
#model.add(Dense(256, activation='relu'))
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(.5))
#model.add(BatchNormalization())
#model.add(Dense(num_classes, activation='softmax'))        
        
#model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#t=time.time()
#historytemp = model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=500, epochs=epochs,verbose=1, validation_data=(x_test, y_test))
#print('Training time: %s' % (t - time.time()))
#sample code:
#stacked_pred = np.column_stack ((predict3, predict2, predict3))
#meta_model = LinearRegression()
#meta_model.fit(stacked_pred, y_test)
#final_pred = meta_model.predict(stacked_pred)
#InceptionV3
#from keras.models import Model
#from keras.optimizers import Adam
#from keras.layers import GlobalAveragePooling2D
#from keras.layers import Dense
#from keras.applications.inception_v3 import InceptionV3

# Get the InceptionV3 model so we can do transfer learning
#base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
                             
# Add a global spatial average pooling layer
#out = base_inception.output
#out = GlobalAveragePooling2D()(out)
#out = Dense(512, activation='relu')(out)
#out = Dense(512, activation='relu')(out)
#total_classes = 2
#predictions = Dense(2, activation='softmax')(out)

#incmodel = Model(inputs=base_inception.input, outputs=predictions)

#for layer in base_inception.layers:
    #layer.trainable = False
    
# Compile 
#incmodel.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy']) 
#incmodel.summary()


data = [['Simple CNN 2conv2d', 12,71.3, 73.3, 76 ], ['Simple CNN 2conv2d', 100,77.5, 78.1, 81.5],
       ['Simple CNN 3conv2d', 12, 80.8, 82.6, 83],['Simple CNN 4conv2d', 12, 88.2, 88, 90.9], ['Simple CNN 4conv2d', 100, 87.2, 82, 86],
       ['Final CNN 5conv2d', 20, 89.5, 89.6, 89.7],['Final CNN 5conv2d', 100, 89.2, 89.6, 89.7],
       ['Final CNN + Imgaugm', 20, 85, 'na', 'na'],['Final CNN + Imgaugm', 30, 88.9, 'na', 'na'], ['Final CNN + Imgaugm', 100, 89.9, 91.9, 89.3],
       ['Final CNN + Imgaugm', 200, 92.5, 90.8, 92.5],['BEST ATTEMPT Final CNN + Imgaugm', 450, 93.4, 91.7, 95.3],
       ['ResNet + Imgaugm', 10, 61, 'na', 'na'],['ResNet + Imgaugm', 25, 65, 'na', 'na'],['ResNet + Imgaugm', 100, 66, 'na', 'na'],
       ['Stack CNN + Imgaugm + Log Regr in Linear Regression', 30, 92.2, 88.5, 91.2], ['Stack CNN + Imgaugm + InceptionV3 in Linear Regression', "30+50", 92.5, 90.2, 92.2],
       ['Stack CNN + Imgaugm + InceptionV3 in Logistic Regression', '30+50+30', 77, 'na', 'na']]
        
        
pd.DataFrame(data, columns=["CNN Variation", "# of Epochs","Score in Notebook", "Public Score", "Private Score"])