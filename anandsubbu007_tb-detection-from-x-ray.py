import os, cv2 # to import directory of file

import zipfile

import pandas as pd #libraries to read

import numpy as np #for algebric function

import matplotlib.pyplot as plt # for visualization

import matplotlib.image as imgplt #for image visualization

import seaborn as sns #Seaborn for visualization



# Sklearn Libraries

from sklearn.model_selection import train_test_split #for test & train Split

from sklearn.metrics import confusion_matrix, accuracy_score # metrics

from sklearn.preprocessing import StandardScaler #for scaling to increase computing speed



#keras libraries

import tensorflow as tf

import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Conv3D, Flatten, MaxPool2D, Dropout, Activation, AvgPool2D

from keras.preprocessing.image import img_to_array, ImageDataGenerator, load_img

from keras.callbacks import ReduceLROnPlateau

from keras.optimizers import RMSprop, Adam



import warnings

warnings.filterwarnings("ignore")
fdirl=[]

s_id = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        fdir = os.path.join(dirname, filename)

        s_idd = os.path.join(filename)

        fdirl.append(fdir)

        s_id.append(s_idd)

fdirl = fdirl[1:] 

fdirl[:2]
nRowsRead = 1000 # specify 'None' if want to read whole file

# jaypee_metadata.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('../input/chest-xrays-tuberculosis-from-india/jaypee_metadata.csv', delimiter=',', nrows = nRowsRead)

#df1.dataframeName = 'jaypee_metadata.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df[:5]
print(df["findings"].unique())

df[:2]
sns.countplot(df["findings"])

#data is equally distributed
dft = df[df["findings"]=="Tuberculosis"]

dff = df[df["findings"]=="False"]

print(dft.shape)

print(dff.shape)
tt = dft["study_id"].to_list()

ff = dff["study_id"].to_list()
lable = []

for i in s_id:

  if i in tt:

     lable.append(1)

  if i in ff:

      lable.append(0)

lable = np.array(lable)

len(lable)
def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (250, 250), interpolation=cv2.INTER_CUBIC)
train = np.array([read_image(i) for i in fdirl])

print("Train :",train.shape)
print("Count of TB pat  :",list(lable).count(1))

print("Count of False   :",list(lable).count(0))
plt.imshow(train[7][:,:,0])

plt.title(lable[7])

#lable 0 --> No TB

#lable 1 --> TB
x=train

y=lable

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=7)

#y_train = y_train.reshape(-1,1)

#y_test = y_test.reshape(-1,1)

print(x_train.shape)

print(x_test.shape)

print(y_train.shape)

print(y_test.shape)

int_sh = x_train.shape[1:]

print("Input shape :",int_sh)
def tbornot():

  model = Sequential()



  model.add(Conv2D(32,(3, 3), padding='same', input_shape=(250,250,3)))

  model.add(Activation("relu"))

  model.add(MaxPool2D(pool_size=(2, 2)))



  model.add(Conv2D(64,(3,3)))

  model.add(Activation("relu"))

  model.add(MaxPool2D(pool_size=(2, 2)))



  model.add(Conv2D(128,(3,3)))

  model.add(Activation("relu"))

  model.add(Conv2D(250,(3,3)))

  model.add(Activation("relu"))

  #model.add(Dropout(0.25))

  model.add(Conv2D(120,(3,3)))

  model.add(Activation("relu"))

  model.add(AvgPool2D(2,2))

  model.add(Conv2D(64,(3,3)))

  model.add(Activation("relu"))

  model.add(AvgPool2D(2,2))



  model.add(Conv2D(32,(2,2)))

  model.add(Activation("relu"))

  model.add(MaxPool2D(2,2))



  model.add(Flatten())

  model.add(Dense(32))

  model.add(Dropout(0.25))

  model.add(Dense(1))

  model.add(Activation("sigmoid"))



  opt = keras.optimizers.Adam(0.001)

  model.compile(loss="binary_crossentropy",optimizer="Adam",metrics=["accuracy"])

  return model
model = tbornot()
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=4, min_lr=0.0001)
datagen = ImageDataGenerator(

    shear_range=0.1,

    zoom_range=0.2,

    rescale=1.0/255.0,

    rotation_range = 10)

datagen.fit(x_train)
mod2 = model.fit(datagen.flow(x_train,y_train,batch_size=30),

                validation_data=(x_train,y_train),epochs=100,callbacks=[reduce_lr])
print(mod2.history.keys())
# summarize history for Loss

plt.plot(mod2.history["loss"])

plt.plot(mod2.history["val_loss"])

plt.title("Model Loss")

plt.ylabel('Loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for accuracy

plt.plot(mod2.history['accuracy'])

plt.plot(mod2.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
y_pred = model.predict(x_test)

y_pred[:5]
print(accuracy_score(y_pred.round(),y_test))

cm = confusion_matrix(y_pred.round(),y_test,labels=[0,1])

cm
df_cm = pd.DataFrame(cm, index = [0,1],

                  columns = [0,1])

sns.heatmap(df_cm, annot=True)