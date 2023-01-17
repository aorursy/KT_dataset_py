import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

import seaborn as sns

from matplotlib.pyplot import imshow



from keras.backend import clear_session



from keras import applications

from keras import layers

from keras.layers import Flatten, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout  

from keras.models import Sequential, Model, load_model  

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

import keras.backend as K  

from keras.callbacks import ModelCheckpoint  

from keras.callbacks import EarlyStopping



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator



clear_session()



# Any results you write to the current directory are saved as output.



## funtions

   

def graph_acc_loss(model):

    

    sns.set(style = 'darkgrid')

    plt.figure(figsize = (24, 8))

    plt.subplot(2, 2, 1)

    #plt.plot(range(100), model.history['acc'])

    plt.plot(model.history['acc'],'r')  

    plt.plot(model.history['val_acc'],'g')  

    plt.ylabel('TRAINING ACCURACY')

    plt.title('TRAINING ACCURACY vs EPOCHS')

    plt.legend(['train','validation'])

    

    plt.subplot(2, 2, 2)

    plt.plot(model.history['loss'],'r')  

    plt.plot(model.history['val_loss'],'g')  

    plt.ylabel('TRAINING LOSS')

    plt.title('TRAINING LOSS vs EPOCHS')

    plt.legend(['train','validation'])

    

    plt.subplot(2, 2, 3)

    plt.plot(model.history['categorical_accuracy'],'b')  

    plt.xlabel('EPOCHS')

    plt.ylabel('TRAINING CATEGORICAL ACCURACY')

    plt.title('TRAINING CATEGORICAL ACCURACY vs EPOCHS')

    plt.legend(['categorical_accuracy'])

    

    plt.subplot(2, 2, 4)

    plt.plot(model.history['categorical_crossentropy'],'b')  

    plt.xlabel('EPOCHS')

    plt.ylabel('TRAINING CATEGORICAL CROSSENTROPY')

    plt.title('TRAINING CATEGORICAL CROSSENTROPY vs EPOCHS')

    plt.legend(['categorical_crossentropy'])

    

    

def prepare_data(df,width,heigth, channel):

    n_of_images = df.shape[0]

    channel = 3

    # preparing X numpy array with the images content

    #X = np.zeros((15697,48,48,3))

    X = np.zeros((n_of_images,width,heigth, channel))

    count = 0

    

    for file in df['Image']:

        img = image.load_img('../input/train/%s' % file,target_size=(width,heigth, channel))

        x = image.img_to_array(img)

        x = preprocess_input(x)

        

        X[count] = x

        

        if(count%4000==0):

            print("ProcessingImage : " , count+1,", ",file)

        count += 1

    

    # preparing Y numpy with de name of files , labelencoded and onehot encoded apply

    y_encoded = df['Id'].values

    values = np.array(y_encoded)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    #print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    #print(onehot_encoded)

    y = onehot_encoded

   

    # split dataset in 20% validate and rest to train

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    

    return X_train, X_test, y_train, y_test, integer_encoded



def prepare_labels(y):

    values = np.array(y)

    label_encoder = LabelEncoder()

    integer_encoded = label_encoder.fit_transform(values)

    # print(integer_encoded)



    onehot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    # print(onehot_encoded)



    y = onehot_encoded

    # print(y.shape)

    return y, label_encoder



def prepareImages(data, m, dataset):

    print("Preparing images")

    X_train = np.zeros((m, 96, 96, 3))

    count = 0

    

    for fig in data['Image']:

        #load images into images of size 100x100x3

        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(96, 96, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_train
df_ = pd.read_csv('../input/train.csv', encoding='utf8')

print(df_['Id'].describe())
df = df_.loc[df_['Id'] != 'new_whale']

number_of_clases = len(df["Id"].value_counts())

print ("Number of Classes: %s" % number_of_clases)
train = df

counted = train.groupby("Id").count().rename(columns={"Image":"image_count"})

counted.loc[counted["image_count"] > 60,'image_count'] = 60

plt.figure(figsize=(25,4))

sns.countplot(data=counted, x="image_count")

plt.show()
X_train, X_test, y_train, y_test, integer_encoded = prepare_data(df,96,96,3)
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

from keras.optimizers import Adam

from keras.applications import MobileNet

from keras.applications.mobilenet import preprocess_input



model = MobileNet(input_shape=(96, 96, 3), alpha=1., weights=None, classes=5004)

model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy',

              metrics=['acc','mse',categorical_crossentropy, categorical_accuracy])

#model.summary()
mobilenet = model.fit(x=X_train/255, y=y_train, epochs=600, batch_size=100, verbose=1, validation_data=(X_test/255, y_test), shuffle=True)
#print("Baseline Error: %.2f%%" % (100-scores[4]*100)) 
graph_acc_loss(mobilenet)
##Save partly trained model 

model.save('00_mobilenet_trained.h5')
# Feature Standardization

def augmentation_feature_standardization(X_train):

    X_train_clone = X_train

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    datagen.fit(X_train_clone)

    return X_train_clone



# Random Flips

def augmentation_random_flips(X_train):

    X_train_clone = X_train

    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    datagen.fit(X_train_clone)

    return X_train_clone



# Random Rotations

def augmentation_random_rotations(X_train):

    X_train_clone = X_train

    datagen = ImageDataGenerator(rotation_range=90)    

    datagen.fit(X_train_clone)

    return X_train_clone



# Random shifts

def augmentation_random_shifts(X_train):

    X_train_clone = X_train

    datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)    

    datagen.fit(X_train_clone)

    return X_train_clone



# ZCA whitening

def augmentation_random_zca(X_train):

    X_train_clone = X_train

    datagen = ImageDataGenerator(zca_whitening=True)   

    datagen.fit(X_train_clone)

    return X_train_clone
X_train_featureStandarization = augmentation_feature_standardization(X_train)

mobilenet1 = model.fit(x=X_train_featureStandarization/255, y=y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_test/255, y_test), shuffle=True)
##Save partly trained model 

model.save('01_mobilenet_featuredStandarization_trained.h5')
graph_acc_loss(mobilenet1)
X_train_randomFlips = augmentation_random_flips(X_train)

mobilenet2 = model.fit(x=X_train_randomFlips/255, y=y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_test/255, y_test), shuffle=True)
##Save partly trained model 

model.save('02_mobilenet_randomFlips_trained.h5')
graph_acc_loss(mobilenet2)
X_train_randomRotations = augmentation_random_rotations(X_train)

mobilenet3 = model.fit(x=X_train_randomRotations/255, y=y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_test/255, y_test), shuffle=True)
##Save partly trained model 

model.save('03_mobilenet_randomRotation_trained.h5')
graph_acc_loss(mobilenet3)
X_train_randomShifts = augmentation_random_shifts(X_train)

mobilenet4 = model.fit(x=X_train_randomShifts/255, y=y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_test/255, y_test), shuffle=True)
##Save partly trained model 

model.save('04_mobilenet_randomShifts_trained.h5')
graph_acc_loss(mobilenet4)
#X_train_randomZca = augmentation_random_zca(X_train)

#mobilenet5 = model.fit(x=X_train_randomZca/255, y=y_train, epochs=50, batch_size=100, verbose=1, validation_data=(X_test/255, y_test), shuffle=True)

#graph_acc_loss(mobilenet)
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):

    # create a grid of 3x3 images

    for i in range(0, 9):

        plt.subplot(330 + 1 + i)

        #plt.imshow(X_batch[i].reshape(96, 96), cmap=plt.get_cmap('gray'))

        #plt.imshow(X_batch[i], cmap=plt.get_cmap('gray'))

        plt.imshow(X_batch[i])

    # show the plot

    plt.show()

    break
##Save partly trained model 

#model.save('augmentation_mobilenet_trained.h5') 

#del model 

##Reload model 

#model = load_model('partly_trained.h5') 
test = os.listdir("../input/test/")

print(len(test))

col = ['Image']

test_df = pd.DataFrame(test, columns=col)



test_df['Id'] = ''

X = prepareImages(test_df, test_df.shape[0], "test")

X /= 255
y_, label_encoder = prepare_labels(df['Id'])
prediction = model.predict(np.array(X), verbose=1)
for i, pred in enumerate(prediction):

    #print (pred)

    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_df.to_csv('submission_v2.csv', index=False)
test_df.head(10)