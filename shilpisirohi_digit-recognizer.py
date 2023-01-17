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
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split





import tensorflow as tf

from tensorflow.keras import models, layers, optimizers

from tensorflow.keras.utils import to_categorical
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df =  pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.head()
test_df.head()
# Displaying the few training data images

def display_data(train_df,grid_size=3):

    train_data = train_df.drop('label',axis=1)



    grid_size=grid_size



    fig ,ax = plt.subplots(nrows=grid_size,ncols=grid_size)



    for row in range(grid_size):

        for col in range(grid_size):

            ax[row,col].imshow(train_data.iloc[grid_size*row+col].values.reshape((28,28)), cmap=plt.get_cmap("gray"))

        

#calling function

display_data(train_df,grid_size=5)
# divide data into train and validate dataset



train,validate = train_test_split(train_df,test_size=0.3, random_state=100)

train_Y, val_Y = train.pop('label'),validate.pop('label')

train_X, val_X = train.values,validate.values
# printing shapes

train_X.shape, val_X.shape, train_Y.shape, val_Y.shape
# reshaping the images 

train_X = train_X.reshape((train_X.shape[0],28*28))

val_X = val_X.reshape((val_X.shape[0],28*28))

test_X = test_df.values.reshape((test_df.shape[0],28*28))



train_X = train_X/255.

val_X = val_X/255.

test_X = test_X/255.





#one-hot encoding for the labels

train_Y = to_categorical(train_Y,num_classes=10)

val_Y = to_categorical(val_Y, num_classes=10)

print(train_X.shape, val_X.shape, train_Y.shape, val_Y.shape,test_X.shape)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(1000,activation='relu', input_shape=(784,)))

model.add(tf.keras.layers.Dense(500, activation='relu'))

model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()
# compile the model

loss=tf.keras.losses.CategoricalCrossentropy()

optimizer=optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# train the model

tf.keras.backend.clear_session()

num_epochs=10

batch_size=32

history=model.fit(x=train_X,y=train_Y,validation_data=(val_X,val_Y), epochs=num_epochs, batch_size=batch_size)
# diagram of the model

tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
# visualizating accuracy and loss



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
res= model.predict(test_X)
pred = np.argmax(res, axis=1)

print(pred.shape)
df_final = pd.DataFrame(columns = ['ImageId','Label'])

df_final['Label'] = pred

df_final['ImageId'] = df_final.index + 1

# np.savetxt('final_1.csv',pred)

df_final.to_csv("final_2.csv")