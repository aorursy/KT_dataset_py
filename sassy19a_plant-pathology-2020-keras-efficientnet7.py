!pip install -U efficientnet
import numpy as np

import pandas as pd

import os

import cv2



import efficientnet.tfkeras as efn

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'



folder = '/kaggle/input/plant-pathology-2020-fgvc7'
Y = pd.read_csv(os.path.join(folder,'train.csv'))

train_id = Y['image_id']

Y = Y[Y.columns[1:]] # remove image_id column

Y = Y.values # convert to array
def image_resize(size, img_id):

    '''

    resize all images to same dimensions

    amended from https://www.kaggle.com/shawon10/plant-pathology-classification-using-densenet121

    '''

    images=[]

    for i, name in enumerate(img_id):

        path=os.path.join(folder,'images',name+'.jpg')

        img=cv2.imread(path)

        image=cv2.resize(img,(size,size),interpolation=cv2.INTER_AREA)

        images.append(image)

        # print processing counter

        if i%200==0:

            print(i, 'images processed')

    return images





X = image_resize(100, train_id)

X = np.array(X)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
aug = ImageDataGenerator(rotation_range=360, # Degree range for random rotations

                          width_shift_range=0.2, # Range for random horizontal shifts

                          height_shift_range=0.2, # Range for random vertical shifts

                          zoom_range=0.2, # Range for random zoom

                          horizontal_flip=True, # Randomly flip inputs horizontally

                          vertical_flip=True) # Randomly flip inputs vertically



train_flow = aug.flow(X_train, Y_train, batch_size=32)
def model(input_shape, classes):

    '''

    transfer learning from imagenet's weights, using Google's efficientnet7 architecture

    top layer (include_top) is removed as the number of classes is changed

    '''

    base = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)



    model = Sequential()

    model.add(base)

    model.add(GlobalAveragePooling2D())

    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',

                optimizer='adam',

                metrics=['accuracy'])

    return model



# each pic has been resized to 100x100, and with 3 channels (RGB)

input_shape = (100,100,3)

classes = 4

model = model(input_shape, classes)

model.summary()
# for every epoch, the total original images will be augmented randomly

model.fit_generator(train_flow,

                    steps_per_epoch=32,

                    epochs=15,

                    verbose=1,

                    validation_data=(X_val, Y_val),

                    use_multiprocessing=True,

                    workers=2)
def plot_validate(model, loss_acc):

    '''

    Plot model accuracy or loss for both train and test validation per epoch

    model : fitted model

    loss_acc : input 'loss' or 'acc' to plot respective graph

    '''

    history = model.history.history



    if loss_acc == 'loss':

        axis_title = 'loss'

        title = 'Loss'

        epoch = len(history['loss'])

    elif loss_acc == 'acc':

        axis_title = 'accuracy'

        title = 'Accuracy'

        epoch = len(history['loss'])



    plt.figure(figsize=(15,4))

    plt.plot(history[axis_title])

    plt.plot(history['val_' + axis_title])

    plt.title('Model ' + title)

    plt.ylabel(title)

    plt.xlabel('Epoch')



    plt.grid(b=True, which='major')

    plt.minorticks_on()

    plt.grid(b=True, which='minor', alpha=0.2)



    plt.legend(['Train', 'Test'])

    plt.show()





plot_validate(model, 'acc')

plot_validate(model, 'loss')
# read in test images & resize

Y_test = pd.read_csv(os.path.join(folder,'test.csv'))

test_id = Y_test['image_id']

X_test = image_resize(100, test_id)

X_test = np.array(X_test)

print('Test images done')
# get prediction probabilities for each class

predict_prob = model.predict(X_test)



df_predict_prob = pd.DataFrame(predict_prob, columns=['healthy','multiple_diseases','rust','scab'])



# join both image_id & df_predict_prob together for submission

frame = [test_id, df_predict_prob]

df_submission = pd.concat(frame, axis=1)

df_submission.to_csv(r'submisson.csv', index=False)

# df_submission.to_csv(r'/kaggle/working/submisson.csv', index=False)