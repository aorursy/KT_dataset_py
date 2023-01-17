import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
#Displaying directory files
os.listdir('/kaggle/input/lego-minifigures-classification/')
#Initializing variable with path to dataset
dataPath = '../input/lego-minifigures-classification/'
#Reading image -> Resizing to (512x512) -> Converting to RGB -> Normalizing pixel values

image = cv2.imread('../input/lego-minifigures-classification/harry-potter/0002/009.jpg')
image = cv2.resize(image, (512,512))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

plt.imshow(image)
#Reading index.csv dataframe
index_df = pd.read_csv(dataPath + 'index.csv')
index_df
#Reading metadata.csv dataframe
meta_df = pd.read_csv(dataPath + 'metadata.csv')
meta_df
#Merging index and metadata on class_id feature
data_df = pd.merge(index_df, meta_df[['class_id', 'minifigure_name']], on='class_id')
data_df
#Displaying overall information
data_df.info()
#Looking for missing data in dataframe
print("Missing Data:",data_df.isnull().any().any())
data_df.isnull().sum()
#Keeping count of number of each minifigure
labels = data_df['minifigure_name'].unique()
count = data_df['minifigure_name'].value_counts()

count
#Visualizing quantity of each minifugure in dataset
import seaborn as sns

plt.figure(figsize=(12,10))
sns.barplot(x=labels, y=count,palette="rocket")

plt.xticks(rotation= 90)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Dataset Analysis')
plt.show()
#Splitting and creating a training and validation dataframe 
training = data_df[data_df["train-valid"] == 'train']
validation = data_df[data_df["train-valid"] == 'valid']
#Viewing our prepared dataframes for training and validation
training, validation
#Evaluating total number of classes
CLASSES = len(data_df['class_id'].unique())
CLASSES
#Training Data Preprocessing

trainData = np.zeros((training.shape[0], 512, 512, 3))

for i in range(training.shape[0]):
    
    image = cv2.imread('../input/lego-minifigures-classification/' + training["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (512,512))
    
    #Normalizing pixel values to [0,1]
    trainData[i] = image / 255.0

trainLabel = np.array(training["class_id"])-1
#Validation Data Preprocessing

validData = np.zeros((validation.shape[0], 512, 512, 3))

for i in range(validation.shape[0]):
    
    image = cv2.imread('../input/lego-minifigures-classification/' + validation["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (512,512))
    
    #Normalizing pixel values to [0,1]
    validData[i] = image / 255.0

validLabel = np.array(validation["class_id"])-1
#Viewing our prepared numpy arrays of data and labels
trainData, trainLabel
#Loading Base Model
base_model = MobileNetV2()

#Adding Dropout layer
x = Dropout(0.5)(base_model.layers[-2].output)

#Adding Dense layer
outputs = Dense(CLASSES, activation='softmax')(x)

#Creating model
model = Model(base_model.inputs, outputs)
#Displaying model summary
model.summary()
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
#Training model - 30 epochs
hist = model.fit(
    trainData, trainLabel,
    epochs=40,
    validation_data=(validData, validLabel),
    shuffle=True,
    batch_size=4
)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='valid loss')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist.history['accuracy'], label='train acc')
plt.plot(hist.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend()
testImage = cv2.imread('../input/lego-minifigures-classification/marvel/0001/001.jpg')
testImage = cv2.resize(testImage, (512,512))
testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2RGB) / 255.0

plt.imshow(testImage)
testImage = np.reshape(testImage, (1, 512, 512, 3))

predictedClass = model.predict(testImage).argmax()
predictedClass = predictedClass + 1

figureName = meta_df['minifigure_name'][meta_df['class_id'] == predictedClass].iloc[0]

print(figureName)