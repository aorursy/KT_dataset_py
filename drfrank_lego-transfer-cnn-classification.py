# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# libraries
import numpy as np 
import pandas as pd

import tensorflow as tf
from tensorflow .keras.models import Sequential 
from tensorflow .keras.layers import Flatten,Dropout, Dense

from keras.preprocessing.image import  load_img,img_to_array

import matplotlib.pyplot as plt
from glob import glob
import os
path = '../input/lego-minifigures-classification/'
jurassic_world_path = "jurassic-world/"
img = load_img(path+jurassic_world_path + "0001/001.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()
x=img_to_array(img)
print(x.shape)
index_df = pd.read_csv('../input/lego-minifigures-classification/index.csv')
index_df.drop('Unnamed: 0', axis=1, inplace=True)
index_df.head(2)
meta_df = pd.read_csv(path+'metadata.csv')
meta_df.head(2)
data_df = pd.merge(index_df, meta_df[['class_id', 'minifigure_name']], on='class_id')
data_df.head(2)
sample_df=data_df.sample(20)
sample_df.head(5)
plt.figure(figsize=(10,10))
i=0
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.grid(False)
    img=load_img('../input/lego-minifigures-classification/'+sample_df['path'].values[i])
    plt.imshow(img)
    #plt.axis("off")
    plt.xlabel(sample_df['minifigure_name'].values[i])
    i += 1
plt.show()
data_df.info()
data_df.isnull().sum()
import plotly.graph_objects as go
df_minifigure_name=data_df['minifigure_name'].value_counts().to_frame().reset_index().rename(columns={'index':'minifigure_name','minifigure_name':'Count'})

fig = go.Figure(go.Bar(
    x=df_minifigure_name['minifigure_name'],y=df_minifigure_name['Count'],
    marker={'color': df_minifigure_name['Count'], 
    'colorscale': 'agsunset'},  
    text=df_minifigure_name['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Minifigure Count',xaxis_title="Minifigure Name",yaxis_title="Count",title_x=0.5)
fig.show()
# Training and Validation Dataframe 

train_set = data_df[data_df["train-valid"] == 'train']

validation_set = data_df[data_df["train-valid"] == 'valid']
import cv2

#We converted the pixels of the image data to array

# Training Data Preprocessing

train_Data = np.zeros((train_set.shape[0], 512, 512, 3))

for i in range(train_set.shape[0]):
    
    image = cv2.imread('../input/lego-minifigures-classification/' + train_set["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (512,512))
    
    #Normalizing pixel values to [0,1]
    train_Data[i] = image / 255.0

trainLabel = np.array(train_set["class_id"])-1
#Validation Data Preprocessing

valid_Data = np.zeros((validation_set.shape[0], 512, 512, 3))

for i in range(validation_set.shape[0]):
    
    image = cv2.imread('../input/lego-minifigures-classification/' + validation_set["path"].values[i])
    
    #Converting BGR to RGB 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Resizing image to (512 x 512)
    image = cv2.resize(image, (512,512))
    
    #Normalizing pixel values to [0,1]
    valid_Data[i] = image / 255.0

validLabel = np.array(validation_set["class_id"])-1
print('Train Label: ',trainLabel.shape)
print('Train Data: ',train_Data.shape)
print('Valid Data: ',valid_Data.shape)
print('Valid Label: ',validLabel.shape)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
dense_net = tf.keras.applications.DenseNet121()
dense_net_layer=Dropout(0.5)(dense_net.layers[-2].output)
number_of_classes = len(data_df['class_id'].unique())
last_layer = Dense(number_of_classes, activation="softmax")(dense_net_layer)
model = Model(dense_net.input, last_layer)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(0.0001),
              metrics=['accuracy'])
print(model.summary())
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='model.h5', monitor="val_accuracy", save_best_only=True, verbose=1)
hist=model.fit(
    train_Data, 
    trainLabel, 
    epochs=50, 
    validation_data=(valid_Data, validLabel), 
    shuffle=True, 
    batch_size=4, 
    callbacks=checkpoint
)
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()

from tensorflow.keras import models as tf_models
# Load the best model (we create for checkpoint to save the best model)
model = tf_models.load_model('model.h5')
sample_df=data_df.sample(40)

from sklearn.model_selection import train_test_split

test, _ = train_test_split(sample_df, test_size=0.5)
test
for i in range(20):
    
    image = cv2.imread('../input/lego-minifigures-classification/'+test['path'].values[i])
    image = cv2.resize(image, dsize=(512,512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
    plt.imshow(image)
    plt.xlabel(test['minifigure_name'].values[i]+'--'+str(test['class_id'].values[i]))
    image = np.reshape(image, (1, 512, 512, 3))
    ans = model.predict(image).argmax()
    ans = ans+1
    minifigure = meta_df["minifigure_name"][meta_df["class_id"] == ans].iloc[0]
    print("Class:", str(ans)+ " Minifigure:",minifigure)
    plt.show()
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from matplotlib import pyplot
from keras.preprocessing import image
img = load_img('../input/lego-minifigures-classification/marvel/0004/008.jpg')

data = img_to_array(img)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(rotation_range=120)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    pyplot.imshow(image)
pyplot.show()
train_set.info()
test_df=train_set.copy()
test_df=test_df.sample(5)
test_df.head()
indexs=test_df.index
for i in indexs:
    train_set.drop(i, axis = 0,inplace = True)

test_df
train_set.info()
batch= 15
size= 256
Epoch= 100
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=20,
                                   width_shift_range=0.4, 
                                   height_shift_range=0.4,
                                   fill_mode="nearest",
                                   zoom_range=0.4,
                                   vertical_flip=True,
                                   horizontal_flip=True,
                                   brightness_range=[0.2,1.0])
valid_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_dataframe(dataframe=train_set, directory=path,
                                                   x_col='path', y_col='minifigure_name', batch_size= batch,
                                                   shuffle=True, target_size=(size,size))
valid_generator = valid_datagen.flow_from_dataframe(dataframe=validation_set, directory=path,
                                                   x_col='path', y_col='minifigure_name', batch_size= batch,
                                                   shuffle=False, target_size=(size,size))
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
dense_net = tf.keras.applications.DenseNet121()

dense_net_layer=Dropout(0.5)(dense_net.layers[-2].output)
number_of_classes = len(data_df['class_id'].unique())-1
last_layer = Dense(number_of_classes, activation="softmax")(dense_net_layer)
model1 = Model(dense_net.input, last_layer)
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model1.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(filepath='model1.h5', monitor="val_accuracy", save_best_only=True, verbose=1)
hist1 = model1.fit_generator(train_generator,
                            epochs=Epoch,
                            validation_data=valid_generator,
                            callbacks=checkpoint)
print(hist1.history.keys())
plt.plot(hist1.history["loss"], label = "Train Loss")
plt.plot(hist1.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist1.history["accuracy"], label = "Train acc")
plt.plot(hist1.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()

test_df
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_dataframe(dataframe=test_df, directory=path, x_col='path', y_col='minifigure_name', batch_size= 1,
                                 shuffle=False, target_size=(size,size))
model1.evaluate_generator(generator=valid_generator)
test_generator.reset()
pred=model1.predict_generator(test_generator,verbose=1)
predicted_classes=np.argmax(pred,axis=1)
predicted_classes
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_classes]
labels
filenames=test_generator.filenames
results=pd.DataFrame({"path":filenames,
                      "Predictions":predictions})
results
final_result = pd.merge(test_df[['minifigure_name','path']], results[['path', 'Predictions']], on='path')
final_result
for i in range(5):
    
    image = cv2.imread('../input/lego-minifigures-classification/'+final_result['path'].values[i])
    image = cv2.resize(image, dsize=(512,512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255
    plt.imshow(image)
    plt.xlabel(final_result['minifigure_name'].values[i]+'****'+final_result['Predictions'].values[i])
    plt.show()