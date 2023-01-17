import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import os

from keras.preprocessing.image import ImageDataGenerator

from keras import layers 

from keras import models

from keras import optimizers

from keras import callbacks 

import matplotlib.ticker as ticker

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

df_test = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')



print("Dataframe shape (train):",df_train.shape)

print("Dataframe shape (test):",df_test.shape)

df_train.head()
df_test.head()
X_train=df_train.iloc[:, 1:].values # feature

y_train=df_train['label'].values # label

num_train=y_train.shape[0]

n_pixel=int(np.sqrt(X_train.shape[1])) # number of pixels in one axis



X_test=df_test.iloc[:, 1:].values # feature

y_test=df_test['label'].values # label

num_test=y_test.shape[0]



X_train=X_train.reshape(num_train,n_pixel,n_pixel,1)

X_test=X_test.reshape(num_test,n_pixel,n_pixel,1)



print("X_train shape",X_train.shape)

print("y_train shape",y_train.shape)

print("X_test shape",X_test.shape)

print("y_test shape",y_test.shape)

print("Image size: %d x %d"%(n_pixel,n_pixel))
plt.figure(figsize=(14,14))

for j in range(20):

    plt.subplot(4,5,j+1)

    plt.imshow(X_train[j,:,:,0],cmap = 'gray')

    plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)

    plt.text(1,3,"%d"%y_train[j],color='white', fontsize=14, bbox=dict(facecolor='black',alpha=0.8))



plt.subplots_adjust(wspace=0, hspace=-0.5)
print("Number of labels in the training set:",len(df_train['label'].value_counts()))

print("Number of labels in the test set:",len(df_test['label'].value_counts()))



plt.figure(figsize=(14,4))

plt.subplot(121)

plt.bar(df_train['label'].value_counts().index,df_train['label'].value_counts().values, color='royalblue')

plt.ylabel('Number of counts')

plt.xlabel('Label')

plt.title('Training set')



plt.subplot(122)

plt.bar(df_test['label'].value_counts().index,df_test['label'].value_counts().values, color='royalblue')

plt.ylabel('Number of counts')

plt.xlabel('Label')

plt.title('Test set')

plt.show()
print("The mean number of counts in the training labels: %0.1f"%np.mean(df_train['label'].value_counts().values))

print("The standard deviation of the number of counts in the training labels: %0.3f"%np.std(df_train['label'].value_counts().values))



print("The mean number of counts in the test labels:%0.1f"%np.mean(df_test['label'].value_counts().values))

print("The standard deviation of the number of counts in the test labels:%0.3f"%np.std(df_test['label'].value_counts().values))
sign_num = 4

idx = np.where(y_train==sign_num)



plt.figure(figsize=(20,5))

for j in range(10):

    plt.subplot(1,10,j+1)

    plt.imshow(X_train[idx[0][j],:,:,0],cmap = 'gray')

    plt.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)

    plt.text(1,3,"%d"%y_train[idx[0][j]],color='white', fontsize=14, bbox=dict(facecolor='black',alpha=0.8))



plt.subplots_adjust(wspace=0, hspace=-0.5)
X_partial_train, X_val, y_partial_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)



y_partial_train=y_partial_train.reshape(-1,1)

y_val=y_val.reshape(-1,1)



print("Before one-hot encoding..")

print("X_partial_train shape:", X_partial_train.shape)

print("y_partial_train shape:", y_partial_train.shape)

print("X_val shape:", X_val.shape)

print("y_val shape:", y_val.shape)



from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

enc.fit(y_partial_train)



y_partial_train=enc.transform(y_partial_train).toarray()

y_val=enc.transform(y_val).toarray()



print("\nAfter one-hot encoding..")

print("y_partial_train shape:", y_partial_train.shape)

print("y_val shape:", y_val.shape)
b_size=64



train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    rotation_range=10,

    width_shift_range=0.1, 

    height_shift_range=0.1, 

    shear_range=0.1,

    zoom_range=0.15,

    horizontal_flip=False,

    fill_mode='nearest')



validation_datagen = ImageDataGenerator(

    rescale=1. / 255)



train_generator = train_datagen.flow(

    X_partial_train,

    y_partial_train,

    batch_size=b_size

)



validation_generator = validation_datagen.flow(X_val, y_val, batch_size=32)
model = models.Sequential()

model.add(layers.Conv2D(64,(3,3),activation='relu', kernel_initializer='he_uniform',input_shape = (n_pixel,n_pixel,1)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_uniform'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512,activation='relu',kernel_initializer='he_uniform'))

model.add(layers.Dense(24,activation='softmax',kernel_initializer='he_uniform'))



model.summary()
csv_logger = callbacks.callbacks.CSVLogger("training.csv", separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1) # early stopping

#reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)



model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])



callbacks_list =[

    csv_logger,

    early_stopping

#    reduce_lr

]



history = model.fit_generator(

      train_generator,

      steps_per_epoch=len(X_train) / b_size,

      epochs=70,

      callbacks=callbacks_list,

      validation_data=validation_generator,

      validation_steps=len(X_val) / b_size

)



model.save('model.h5')
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(frameon=False)



plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(frameon=False)



plt.show()
training_df=pd.read_csv("training.csv")

training_df
print("Finale result:\n", training_df.tail(1))
print("Accuracy for the test dataset: ",np.sum(enc.inverse_transform(model.predict(X_test)).flatten() == y_test)/y_test.shape[0]*100)