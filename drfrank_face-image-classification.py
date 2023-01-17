# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
Face_data=pd.read_csv('../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')
df=Face_data.copy()
df.head(2)
df.info()
df.shape
df.isnull().values.any()
df.isnull().sum()
df.describe().T
df[df.duplicated() == True]
fig = go.Figure(data=[go.Histogram(x=df['age'],  # To get Horizontal plot ,change axis 
                                  marker_color="Bisque",
                      xbins=dict(
                      start=0, #start range of bin
                      end=120,  #end range of bin
                      size=10    #size of bin
                      ))])
fig.update_layout(title="Distribution Of Age",xaxis_title="Age",yaxis_title="Counts",title_x=0.5)
fig.show()
df_age=df['age'].value_counts().reset_index().rename(columns={'index':'age','age':'Count'})

fig = go.Figure(go.Bar(
    x=df_age['age'],y=df_age['Count'],
    marker={'color': df_age['Count'], 
    'colorscale': 'Viridis'},  
    text=df_age['Count'],
    textposition = "outside",
))
fig.update_layout(title_text='Age Of Distribution',xaxis_title="Age",yaxis_title=" Count ",title_x=0.5)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(
    y=df['age'],
    boxmean='sd',
    name="Age",
    jitter=0.3,
    pointpos=-1.8,
    boxpoints='outliers', # represent outliers points
    marker_color='rgb(7,40,89)',
    line_color='rgb(7,40,89)'
))
fig.update_layout(title_text='Age Of Distribution',xaxis_title="Age",yaxis_title="Age Count ",title_x=0.5)
fig.show()
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (15, 5)
sns.distplot(df['age'], color = 'cyan')
plt.title('Distribution of Age', fontsize = 20)
plt.show()
print("Avg Age: " + str(df["age"].mean()))
print("Max Age: " + str(df["age"].max()))
print("Min Age: " + str(df["age"].min()))
print("Median Age: " + str(df["age"].median()))


df_ethnicity=df['ethnicity'].value_counts().reset_index().rename(columns={'index':'ethnicity','ethnicity':'count'}).sort_values('count',ascending="False")

fig = go.Figure(go.Bar(y=df_ethnicity['ethnicity'], x=df_ethnicity['count'], # Need to revert x and y axis
                      orientation="h")) # default orentation value is "v" - vertical ,we need to change it as orientation="h"
fig.update_layout(title_text=' Ethnicity  Frequency ',xaxis_title="Count",yaxis_title="Ethnicity",title_x=0.5)
fig.show()
df_ethnicity=df['ethnicity'].value_counts().to_frame().reset_index().rename(columns={'index':'ethnicity','ethnicity':'count'})

colors=['cyan','royalblue','blue','darkblue',"darkcyan"]
fig = go.Figure([go.Pie(labels=df_ethnicity['ethnicity'], values=df_ethnicity['count'])])
fig.update_traces(hoverinfo='label+percent', textinfo='percent+value', textfont_size=15,
                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(title="Ethnicity Distribution",title_x=0.5)
fig.show()
df_gender=df['gender'].value_counts().to_frame().reset_index().rename(columns={'index':'gender','gender':'count'})


fig = go.Figure([go.Pie(labels=df_gender['gender'], values=df_gender['count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=12,insidetextorientation='radial')

fig.update_layout(title="Gender Count",title_x=0.5)
fig.show()
ax = sns.countplot(x="gender", data=df)
plt.ylabel('Count')
plt.xlabel('Gender')
plt.title('Gender Count');
df_rece_gender=df.groupby(by =['gender','ethnicity'])['img_name'].count().to_frame().reset_index().rename(columns={'ethnicity':'Ethnicity','img_name':'Count'})

df_rece_gender['gender']=df_rece_gender['gender'].astype('category')




fig = px.bar(df_rece_gender, x="Ethnicity", y="Count",color="gender",barmode="group",
             
             )
fig.update_layout(title_text='Ethnicity with Gender',title_x=0.5)
fig.show()
df_pixels = df.pixels.str.split(" ").tolist() 
df_pixels = pd.DataFrame(df_pixels, dtype=int)
df_images = df_pixels.values
df_images = df_images.astype(np.float)

print(df_images.shape)
def show(img):
    show_image = img.reshape(48,48)
    
    plt.axis('off')
    plt.imshow(show_image, cmap='gray')
show(df_images[1000])
plt.figure(0, figsize=(12,6))
for i in range(1,13):
  plt.subplot(3, 4, i)
  plt.axis('off')

  image = df_images[i+i*2].reshape(48,48)
  plt.imshow(image, cmap='gray')

plt.tight_layout()
plt.show()
df_pixels = df.pixels.str.split(" ").tolist() 
df_pixels = pd.DataFrame(df_pixels, dtype=int)
df_images = df_pixels.values
df_images = df_images.astype(np.float)

y=df['ethnicity']
y.head()
clas_number=y.unique()
clas_number=len(clas_number)
clas_number
X=df_images

#normalizing pixels data
X=X/255

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=35
)
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
print('X_Train:', X_train.shape)
print('X_Test:', X_test.shape)
print('y_Train:', y_train.shape)
print('y_Test:', y_test.shape)
model = Sequential()

#1. LAYER

model.add(Conv2D(32, 3, data_format="channels_last", kernel_initializer="he_normal", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#2. LAYER

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))


### 3. LAYER
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
 

### Full Connection layer
model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

### Out Layer

model.add(Dense(5))
model.add(Activation('softmax')) 

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()





epochs = 20
batchSize =100
checkpoint = ModelCheckpoint(filepath='model_ethnicity.h5', monitor="val_accuracy", save_best_only=True, verbose=1)
hist = model.fit(X_train, y_train,
                 epochs=epochs,
                 shuffle=True,
                 batch_size=batchSize,
                 validation_data=(X_test, y_test),
                 callbacks=[checkpoint],
                 verbose=2)
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Traning', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist.history['loss'], color ='r', label='Training Loss')
plt.plot(hist.history['val_loss'], color ='b', label='Validation Loss')
plt.legend(loc='upper right')


plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(hist.history['accuracy'], color ='g', label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], color ='m', label='Validation Accuracy')
plt.legend(loc='lower right')

plt.show()

loss, acc = model.evaluate(X_test,y_test,verbose=0)
print('Test loss: {}'.format(loss))
print('Test Accuracy: {}'.format(acc))
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.22, random_state=35
)
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)
print('X_Train:', X_train.shape)
print('X_Test:', X_test.shape)
print('y_Train:', y_train.shape)
print('y_Test:', y_test.shape)
model = Sequential()

#1. LAYER

model.add(Conv2D(32, 3, kernel_initializer="he_normal", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#2. LAYER

model.add(Conv2D(64, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))


### 3. LAYER
model.add(Conv2D(32, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
 

### Full Connection layer
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

### Out Layer

model.add(Dense(1))
model.add(Activation('sigmoid')) 

model.compile(optimizer='sgd',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(filepath='modelgender.h5', monitor="val_accuracy", save_best_only=True, verbose=1)
hist = model.fit(X_train, y_train,
                 epochs=epochs,
                 shuffle=True,
                 batch_size=batchSize,
                 validation_data=(X_test, y_test),
                 callbacks=[checkpoint],
                 verbose=2)
plt.figure(figsize=(14,3))
plt.subplot(1, 2, 1)
plt.suptitle('Traning', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(hist.history['loss'], color ='r', label='Training Loss')
plt.plot(hist.history['val_loss'], color ='b', label='Validation Loss')
plt.legend(loc='upper right')


plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(hist.history['accuracy'], color ='g', label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], color ='m', label='Validation Accuracy')
plt.legend(loc='lower right')

plt.show()
loss, acc = model.evaluate(X_test,y_test,verbose=0)
print('Test loss: {}'.format(loss))
print('Test Accuracy: {}'.format(acc))