import pickle

from pathlib import Path

from skimage import io



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report



# import keras library

import keras



# import Sequential from the keras models module

# ... YOUR CODE FOR TASK 1 ...

from keras.models import Sequential

# import Dense, Dropout, Flatten, Conv2D, MaxPooling2D from the keras layers module

from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
# load labels.csv from datasets folder using pandas

labels = pd.read_csv('../input/sgp-labels/labels.csv',index_col=0)



# print value counts for genus

print(labels.genus.value_counts())



# assign the genus label values to y

y=labels['genus'].values
# load an image and explore

example_image = io.imread('../input/bee-data/datasets/{}.jpg'.format(labels.index[0]))



# show image

plt.imshow(example_image)



# print shape

print('Example image has shape: ', example_image.shape)

                    

# print color channel values for top left pixel

print('RGB values for the top left pixel are:', example_image[0,0,:])
# initialize standard scaler

ss = StandardScaler()



image_list = []

for i in labels.index:

    # load image

    img = io.imread('../input/bee-data/datasets/{}.jpg'.format(i)).astype(np.float64)

    

      

    # for each channel, apply standard scaler's fit_transform method

    for channel in range(img.shape[2]):

        img[:, :, channel] = ss.fit_transform(img[:, :, channel])

        

   

    # append to list of all images

    image_list.append(img)

    



# convert image list to single array

X = np.array(image_list)

# print shape of X

print(X.shape)

imglist=[]

img1=io.imread('../input/finaltest/h4.jpg').astype(np.float64)    

for channel1 in range(img1.shape[2]):

    img1[:, :, channel1] = ss.fit_transform(img1[:, :, channel1])



imglist.append(img1)

l=np.array(imglist)

print(l.shape)
# split out evaluation sets (x_eval and y_eval)

x_interim, x_eval, y_interim, y_eval = train_test_split(X,

                                           y,

                                           test_size=0.2,

                                           random_state=52)



# split remaining data into train and test sets

x_train,x_test,y_train,y_test=train_test_split(x_interim,y_interim,test_size=0.4,random_state=52)

# examine number of samples in train, test, and validation sets

print('x_train shape:', x_train.shape)

print(x_train, 'train samples')

print(x_test.shape[0], 'test samples')

print(x_eval.shape[0], 'eval samples')
# set model constants

num_classes = 1



# define model as Sequential

model = Sequential()



# first convolutional layer with 32 filters

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50, 50, 3)))



# add a second 2D convolutional layer with 64 filters

model.add(Conv2D(64,kernel_size=(3,3,),activation='relu'))
# reduce dimensionality through max pooling

model.add(MaxPooling2D(pool_size=(2,2)))



# third convolutional layer with 64 filters

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

# add dropout to prevent over fitting

model.add(Dropout(0.25))

# necessary flatten step preceeding dense layer

model.add(Flatten())

# fully connected layer

model.add(Dense(128, activation='relu'))



# add additional dropout to prevent overfitting

model.add(Dropout(0.5))



# prediction layers

model.add(Dense(num_classes, activation='sigmoid', name='preds'))



# show model summary

model.summary()
model.compile(

    # set the loss as binary_crossentropy

    loss=keras.losses.binary_crossentropy,

    # set the optimizer as stochastic gradient descent

    optimizer=keras.optimizers.SGD(lr=0.001),

    # set the metric as accuracy

    metrics=['accuracy']

)



# mock-train the model using the first ten observations of the train and test sets

model.fit(

    x_train[:10, :, :, :],

    y_train[:10],

    epochs=5,

    verbose=1,

    validation_data=(x_test[:10, :, :, :], y_test[:10])

)
# load pre-trained model

pretrained_cnn = keras.models.load_model('../input/sgp-labels/pretrained_model.h5')



# evaluate model on test set

score = pretrained_cnn.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])



print("")



# evaluate model on holdout set

eval_score = pretrained_cnn.evaluate(x_eval,y_eval,verbose=0)

# print loss score

print('Eval loss:', eval_score[0])

# print accuracy score

print('Eval accuracy:', eval_score[1])
# load history

with open('../input/sgp-labels/model_history.pkl', 'rb') as f:

    pretrained_cnn_history = pickle.load(f)



# print keys for pretrained_cnn_history dict

print(pretrained_cnn_history)



fig = plt.figure(1)

plt.subplot(211)

# plot the validation accuracy

plt.plot(pretrained_cnn_history['val_acc'])

plt.title('Validation accuracy and loss')

plt.ylabel('Accuracy')

plt.subplot(212)

# plot the validation loss

plt.plot(pretrained_cnn_history['val_loss'], 'r')

plt.xlabel('Epoch')

plt.ylabel('Loss value');
e=pretrained_cnn.predict(l)

print(e)

k=pretrained_cnn.predict_classes(l)

print(k)

if k!=0:

    print("Bumble Bee!")

else:

    print("honey Bee!!")





print("")



# predicted probabilities for x_eval

y_proba=pretrained_cnn.predict(x_eval)



print("First five probabilities:")

print(y_proba[:5])

print("")



# predicted classes for x_eval

# ... YOUR CODE FOR TASK 11 ...

y_pred=pretrained_cnn.predict_classes(x_eval)

print("First five class predictions:")

x=y_pred[:5]

print(x)

for i in x:

     if i==0:

             print("Honey Bee!")

     else:

            print("Bumble Bee!!")

print("")