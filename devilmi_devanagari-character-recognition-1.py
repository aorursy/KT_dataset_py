#Importing Libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

#Reading Data
data=pd.read_csv('../input/data.csv')

data.head()
data.groupby('character').count()
#Visualizing the Characters
char_names = data.character.unique()  

rows =10;columns=5;

fig, ax = plt.subplots(rows,columns, figsize=(8,16))

for row in range(rows):

    for col in range(columns):

        ax[row,col].set_axis_off()

        if columns*row+col < len(char_names):

            x = data[data.character==char_names[columns*row+col]].iloc[0,:-1].values.reshape(32,32)

            x = x.astype("float64")

            x/=255

            ax[row,col].imshow(x, cmap="binary")

            ax[row,col].set_title(char_names[columns*row+col].split("_")[-1])



            

plt.subplots_adjust(wspace=1, hspace=1)   

plt.show()
#Data Preprocessing
#Normalizing the data

X = data.values[:,:-1]/255.0

Y = data["character"].values
from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



# splitting the data into train and test data

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)



# Encoding the categories

le = LabelEncoder()

y_train = le.fit_transform(y_train)

y_test = le.transform(y_test)

y_train=to_categorical(y_train,46)

y_test=to_categorical(y_test,46)
y_train.shape
#Converting 1D data to 2D

im_shape=(32,32,1)

x_train=x_train.reshape(x_train.shape[0],*im_shape)

x_test=np.reshape(x_test,(x_test.shape[0],32,32,1))

x_test.shape
#Model making
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(32,32,1), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))



model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(units=128, activation='relu', kernel_initializer='uniform'))

model.add(Dense(units=64, activation='relu', kernel_initializer='uniform'))

model.add(Dense(units=46, activation='softmax', kernel_initializer='uniform'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train, y_train, batch_size=32, 

                  epochs=10,validation_data=(x_test, y_test))
model.evaluate(x_test,y_test)
print(history.history.keys())
# summarize history for accuracyh5

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.save('model1.h5')