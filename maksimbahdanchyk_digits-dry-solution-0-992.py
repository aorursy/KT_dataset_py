import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
train.label.hist()

train.head()
fig,ax = plt.subplots(5,5,figsize=(10, 10))

for row in range(5):

    for col in range(5):

        rand_row = np.random.randint(0,len(train))

        num = train.iloc[rand_row,1:].values.reshape(28,28)

        ax[row,col].imshow(num,cmap='gray')

        ax[row,col].axis('off')

        ax[row,col].set_title(str(train.iloc[rand_row,0]))

fig.tight_layout() 
X = train.drop('label',axis=1) / 255.0

y = train['label']

print('Train: ',X.shape)

print('Label: ',y.shape)
import tensorflow as tf



X = X.values.reshape(-1,28,28,1)

y = tf.keras.utils.to_categorical(y, num_classes=10, dtype='float32')

print('Train: ',X.shape)

print('Label: ',y.shape)



from sklearn.model_selection import train_test_split, cross_val_score



X_train,X_val,y_train,y_val = train_test_split(X,y,

                                              test_size = 0.1,

                                              shuffle = True,

                                              random_state = 42)

print('Train X shape',X_train.shape)

print('Train y shape',y_train.shape)

print('Val   X shape',X_val.shape)

print('Val   X shape',y_val.shape)
import tensorflow as tf



model = tf.keras.Sequential([

    

    tf.keras.layers.Conv2D(filters = 16, strides=(1, 1),kernel_size = (3,3),

                           padding='same',input_shape=(28,28,1),activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

    

    tf.keras.layers.Conv2D(filters = 32, strides=(1, 1),kernel_size = (3,3),

                           padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

    tf.keras.layers.Dropout(0.1),

    

    tf.keras.layers.Conv2D(filters = 64, strides=(1, 1),kernel_size = (3,3),

                           padding='same',activation='relu'),

    tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='same'),

    tf.keras.layers.Dropout(0.1),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=256,activation='relu'),

    tf.keras.layers.Dense(units=10, activation='softmax')

    

])



model.compile(loss = 'categorical_crossentropy',

              optimizer = 'adam',

              metrics = ['accuracy'])
gen = tf.keras.preprocessing.image.ImageDataGenerator(

        rotation_range=15,

        width_shift_range=0.2,

        height_shift_range=0.2,

        fill_mode="nearest",

)



gen.fit(X_train)



history = model.fit_generator(gen.flow(X_train,y_train, batch_size=32),

                              epochs = 50,

                              validation_data = (X_val,y_val),

                              verbose = 1, 

                              steps_per_epoch=X_train.shape[0] // 32)
history.history
plt.plot(history.history['accuracy'],'b-',label = 'accuracy')

plt.plot(history.history['val_accuracy'],'r-', label  = 'val_accuracy')

plt.legend(loc="upper left")
from sklearn.metrics import plot_confusion_matrix, confusion_matrix



cm = confusion_matrix(y_val.argmax(axis=1),model.predict(X_val).argmax(axis=1))

plt.matshow(cm)

cm

y_val_norm = y_val.argmax(axis=1)

y_pred_norm = model.predict(X_val).argmax(axis=1)

wrong_idx = np.where(np.not_equal(y_val_norm, y_pred_norm))[0]

fig,ax = plt.subplots(4,4)

for row in range(4):

    for col in range(4):

        idx = np.random.choice(wrong_idx)

        ax[row,col].imshow(train.iloc[idx,1:].values.reshape(28,28))

        ax[row,col].set_title(train.iloc[idx,0])

        ax[row,col].axis('off')

fig.tight_layout()

    
imageid = np.arange(1,len(test)+1)

pred = model.predict(test.values.reshape(-1,28,28,1)/255.0)
sub = pd.DataFrame({'ImageId':imageid,'Label':pred.argmax(axis=1)})

sub.to_csv('submission1.csv',index=False)