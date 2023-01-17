import pandas as pd

import tensorflow as tf

import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping



%matplotlib inline
df = pd.read_csv("../input/digit-recognizer/train.csv")
df.describe()
df.head()
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.1,random_state=11)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
y_train.unique()
rows=10 #rows in subplots

cols=10 #columns in subplots

samp = random.sample(range(df.shape[0]),rows*cols) #selecting 100 random samples

df_samp = df.iloc[samp,:].copy()



fig,ax = plt.subplots(rows,cols,figsize=(14,20))

r = 0

c = 0

for i in range(rows*cols):

    arr=np.array(df_samp.iloc[i,1:]).astype(np.int16).reshape(28,28)

    ax[r,c].axis("off")

    ax[r,c].imshow(arr,cmap="gray")

    ax[r,c].set_title(f"Label: {df_samp.iloc[i,0]}")

    c+=1

    if c == cols:

        c=0

        r+=1

plt.show()
X_train = X_train.values.reshape(len(X_train),28,28,1)
X_test = X_test.values.reshape(len(X_test),28,28,1)
train_datagen = ImageDataGenerator(

        rotation_range=20,

        height_shift_range=0.1,

        shear_range=0.2,

        zoom_range=0.2,

        rescale=1./255,

        fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

r = random.randint(0,len(X_train))

ax1.imshow(X_train[r].reshape(28,28),cmap="gray");

ax1.axis("off")

ax1.set_title("Original")

ax2.imshow(train_datagen.random_transform(X_train[r]).reshape(28,28),cmap="gray");

ax2.axis("off")

ax2.set_title("Augmented")

r = random.randint(0,len(X_train))

ax3.imshow(X_train[r].reshape(28,28),cmap="gray");

ax3.axis("off")

ax4.imshow(train_datagen.random_transform(X_train[r]).reshape(28,28),cmap="gray")

ax4.axis("off");
random.seed(11)

cnn = tf.keras.models.Sequential()



cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,data_format='channels_last',activation='relu',input_shape=(28,28,1),padding="same"))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2))) 

cnn.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,strides=1,data_format='channels_last',activation='relu',padding="same"))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

cnn.add(tf.keras.layers.Flatten())





cnn.add(tf.keras.layers.Dense(256,activation="relu"))

cnn.add(tf.keras.layers.Dropout(0.2))

cnn.add(tf.keras.layers.Dense(256,activation="relu"))

cnn.add(tf.keras.layers.Dropout(0.2))

cnn.add(tf.keras.layers.Dense(10,activation="softmax"))
random.seed(11)

cnn.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics="sparse_categorical_accuracy")
early_stop = EarlyStopping(monitor='val_loss',patience=5)
training_data = train_datagen.flow(X_train,y_train,batch_size=32,seed=11)

validation_data = test_datagen.flow(X_test,y_test,batch_size=32,seed=11,shuffle=False)
random.seed(11)

final_model=cnn.fit_generator(training_data,epochs=500,callbacks=[early_stop],validation_data=validation_data)
loss_df = pd.DataFrame(final_model.history)

loss_df.head()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))

loss_df.iloc[:,[0,2]].plot(ax=ax1)

ax1.set_title("Train vs Validation Loss")

loss_df.iloc[:,[1,3]].plot(ax=ax2)

ax2.set_title("Train vs Validation Accuracy")

plt.show()
final_test = pd.read_csv("../input/digit-recognizer/test.csv")
final_test = final_test/255.0
final_test = final_test.values.reshape(len(final_test),28,28,1)
pred = cnn.predict_classes(final_test)
pred
final_submission = pd.DataFrame({"ImageId":range(1,len(pred)+1),"Label":pred})
final_submission
final_submission.to_csv("final_submission.csv",index=False)