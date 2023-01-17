import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import tensorflow as tf

from keras.utils.np_utils import to_categorical

from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split



import os





np.random.seed(92)

# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = '/kaggle/input/digit-recognizer/train.csv'

test_data = '/kaggle/input/digit-recognizer/test.csv'

train_df = pd.read_csv(train_data)

print(train_df.shape)

train_df.head()



test_df = pd.read_csv(test_data)

print(test_df.shape)

test_df.head()
# LET'S SEPARATE THE LABELS FROM THE ACTUAL DATA



if 'label' in train_df.columns:   

    y_train = train_df['label'].values.astype('int32')

#     y_train = train.iloc[:,0].values.astype('int32')

#     y_train = 

    train_df = train_df.drop('label', axis = 1)

else:

    pass



# NOW TO SET UP THE X PART OF THE TRAINING DATA.



x_train = train_df.values.astype('float32')

x_test = test_df.values.astype('float32')
print('the shape of the training data is ;',x_train.shape)

print('the shape for training labels is  :',y_train.shape[0])



print('\nthe shape for the testing data is:', x_test.shape)
# LET'S TAKE A LOOK AT SOME OF THE IMAGES.



#Convert train datset to (num_images, img_rows, img_cols) format 

x_train = x_train.reshape(x_train.shape[0], 28, 28)



for i in range(6, 9):

    plt.subplot(330 + (i+1))

    plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train[i])
x_train = x_train.reshape(x_train.shape[0], 28,28,1)
x_train.shape
x_test  = x_test.reshape(x_test.shape[0],28,28,1)

x_test.shape
y_train.shape
train_max = np.max(x_train)

train_min = np.min(x_train)

# print(train_max)



test_max = np.max(x_test)

test_min = np.min(x_test)

x_train = x_train/255.0

x_test = x_test/255.0
norm_train_max = np.max(x_train)

norm_train_min = np.min(x_train)

# print(train_max)



norm_test_max = np.max(x_test)

norm_test_min = np.min(x_test)
print('The maximum of the training data before normalization was',train_max,'but after normalizing it is', norm_train_max )

print('The maximum of the testing data before normalization was',test_max,'but after normalizing it is', norm_test_max )



print('\nThe minimum of the training data before normalization was',train_min,'but after normalizing it is', norm_train_min )

print('The minimum of the testing data before normalization was',test_min,'but after normalizing it is', norm_test_min )
y_train= to_categorical(y_train)

num_classes = y_train.shape[1]

num_classes

model = tf.keras.models.Sequential([

    

    # 1st BATCH OF COVOLUTIONS 

    tf.keras.layers.Conv2D(16, (5,5),activation='relu', input_shape=(28,28,1)),

    tf.keras.layers.Conv2D(16, (5,5), activation= 'relu'),

    tf.keras.layers.Conv2D(16, (5,5), activation= 'relu'),

    tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

    # 2nd BATCH OF CONVOLUTIONS 

    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),

    tf.keras.layers.Conv2D(32, (3,3), activation= 'relu'),  

    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),

    tf.keras.layers.BatchNormalization(),

# #     tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.25),

    

#     # 3nd BATCH OF CONVOLUTIONS 

#     tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),

#     tf.keras.layers.Conv2D(64, (3,3), activation= 'relu'),  

#     tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),

#     tf.keras.layers.BatchNormalization(),

# #     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Dropout(0.25),

    

    tf.keras.layers.BatchNormalization(),    

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.50),

    

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation = 'relu'),

    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(10, activation='softmax')   

   

   

])



print('input shape  :', model.input_shape)

print('output shape :', model.output_shape)
# VERSION 3



# model = tf.keras.models.Sequential([

    

#    # 1ST BACTCH OF CONVOLUTIONS

    

#     # This is the first convolution batch 

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1), data_format = 'channels_last'),

#     # The second convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),     

#     # The third convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    

#     tf.keras.layers.BatchNormalization(),    

#     tf.keras.layers.Dropout(0.25),

    

#     # 2ND BACTCH OF CONVOLUTIONS

    

#     # This is the first convolution batch 

#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

#     # The second convolution

#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

#     # The third convolution

#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),



#     tf.keras.layers.BatchNormalization(),    

#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Dropout(0.25),

    

#     # 3RD BACTCH OF CONVOLUTIONS

    

#     # This is the first convolution batch 

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

# #     tf.keras.layers.MaxPooling2D(2,2),

#     # The second convolution

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

# #     tf.keras.layers.MaxPooling2D(2,2),

#     # The third convolution

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

    

#     tf.keras.layers.BatchNormalization(),    

#     tf.keras.layers.MaxPooling2D(2,2),

#     tf.keras.layers.Dropout(0.25),

    

#     # Flatten the results to feed into a DNN

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(256),

#     tf.keras.layers.Dropout(0.25),

#     tf.keras.layers.Dense(10, activation='softmax')

# ])



# print('input shape  :', model.input_shape)

# print('output shape :', model.output_shape)
# # VERSION 2



# model = tf.keras.models.Sequential([

    

#    # 1ST BACTCH OF CONVOLUTIONS

    

#     # This is the first convolution batch 

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1), data_format = 'channels_last'),

#     tf.keras.layers.BatchNormalization(),

#     # The second convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),

#     # The third convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.Dropout(0.4),

    

#     # 2ND BACTCH OF CONVOLUTIONS

    

#     # This is the first convolution batch 

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),

#     # The second convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),

#     # The third convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),    

#     tf.keras.layers.Dropout(0.4),

    

#     # 3RD BACTCH OF CONVOLUTIONS

    

#     # This is the first convolution batch 

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),

#     # The second convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),

#     # The third convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

#     tf.keras.layers.BatchNormalization(),    

#     tf.keras.layers.Dropout(0.4),

    

#     # Flatten the results to feed into a DNN

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.BatchNormalization(),

#     tf.keras.layers.Dense(10, activation='softmax')

# ])



# print('input shape  :', model.input_shape)

# print('output shape :', model.output_shape)
# VERSION 1



# model = tf.keras.models.Sequential([

   

#     # This is the first convolution

#     tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),

#     tf.keras.layers.MaxPooling2D(2, 2),

    

#     # The second convolution

#     tf.keras.layers.Conv2D(32, (3,3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2,2),

    

#     # The third convolution

#     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2,2),

    

#     # Flatten the results to feed into a DNN

#     tf.keras.layers.Flatten(),

  

#     tf.keras.layers.Dense(10, activation='softmax')

# ])



# print('input shape  :', model.input_shape)

# print('output shape :', model.output_shape)
model.summary()
### COMPILE THE MODEL

model.compile(loss = 'categorical_crossentropy', optimizer= RMSprop(lr=0.003), metrics = ['acc'])
from keras.preprocessing import image



train_generator = image.ImageDataGenerator()
X = x_train

Y = y_train 



X_train, X_val, Y_train , Y_val = train_test_split(x_train,y_train, test_size= 0.05, random_state = 92)

print(X_train.shape)

batches = train_generator.flow(X_train, Y_train, batch_size=32)

val_batches = train_generator.flow(X_val, Y_val, batch_size=32)
history = model.fit_generator(

      generator=batches, 

      steps_per_epoch=batches.n,

#       steps_per_epoch = 100,

      epochs=20, 

      validation_data=val_batches,

      validation_steps=val_batches.n,

#     validation_steps = 100

)
# history.history['loss']

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training set', 'Validation set'], loc='center left')

plt.show()
predictions = model.predict_classes(x_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)