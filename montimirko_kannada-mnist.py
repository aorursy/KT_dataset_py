import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import tensorflow.keras as keras

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import PIL

from PIL import Image

from tensorflow.keras.applications import DenseNet121



from tensorflow.keras.applications.resnet50 import ResNet50



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
base_path = '/kaggle/input/Kannada-MNIST/'

train = pd.read_csv(base_path + 'train.csv')

test = pd.read_csv(base_path + 'test.csv')

dig_mnist = pd.read_csv(base_path + 'Dig-MNIST.csv')





train_label = train['label']

train_data_raw = train.drop('label', axis=1, inplace=False)

train_data = []



train_data_raw = np.array(train_data_raw)



for i in train_data_raw:

    image = i.reshape(28,28,1)

    train_data.append(image)



#test...

test.drop('id', axis=1, inplace=True)



test_data_raw = np.array(test)



test_data = []

for i in test_data_raw:

    image = i.reshape(28,28,1)

    test_data.append(image)



plt.figure( figsize = (10,10))

for i in range(1,10):

    plt.subplot(3,4,i)

    plt.title(train_label[i])

    plt.imshow(train_data[i].reshape([28,28]))

    

plt.plot()

train_label.value_counts().plot(kind='barh')



#create a np.array of pixel

train_data = np.array(train_data)/255.0

test_data = np.array(test_data)/255.0



train_label  = np.array(train_label).reshape(60000,-1)
es = EarlyStopping(monitor='accuracy', mode='min', verbose=1, patience=5,baseline=0.99)



def define_model():

    model2 = keras.models.Sequential([

        keras.layers.Conv2D(16, (3,3), input_shape=(28,28,1), activation='relu'),#32

        keras.layers.BatchNormalization(),

        #keras.layers.Conv2D(16,(3,3), activation='relu'),

        #keras.layers.BatchNormalization(),

        keras.layers.Conv2D(16, (5,5), activation='relu', padding='same'),

        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(2,2),

        keras.layers.Dropout(0.4),

        keras.layers.Conv2D(32, (3,3), activation='relu'),

        keras.layers.BatchNormalization(),

        #keras.layers.Conv2D(32,(3,3), activation='relu'),

        #keras.layers.BatchNormalization(),

        keras.layers.Conv2D(32, (5,5), activation='relu', padding='same'),

        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(2,2),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),

        keras.layers.Conv2D(64, (3,3), activation='relu'),

        keras.layers.BatchNormalization(),

        #keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),

        #keras.layers.BatchNormalization(),

        keras.layers.Conv2D(64, (5,5), activation='relu', padding='same'),

        keras.layers.BatchNormalization(),

        keras.layers.MaxPooling2D(2,2),

        keras.layers.BatchNormalization(),

        keras.layers.Dropout(0.5),

        keras.layers.Flatten(),

        keras.layers.Dense(128), #256 prima!

        keras.layers.BatchNormalization(),

        keras.layers.Activation('relu'), 

        keras.layers.Dropout(0.6),

        keras.layers.Dense(64),#32 prima

        keras.layers.BatchNormalization(),

        keras.layers.Activation('relu'),

        keras.layers.Dropout(0.5),

        keras.layers.Dense(10, activation='softmax')

    ])

    return model2





model = define_model()



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



model.summary()
def define_model_ResNet50():

    model = keras.models.Sequential()

    input_layer = keras.layers.Input(shape=(224, 224, 3), name='image_input')

    model.add(DenseNet121(weights=None, include_top=False, input_tensor=input_layer))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(32))

    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Dense(10, activation='softmax'))

    

    model.layers[0].trainable = True

    model.summary()

    return model

EPOCHS =100

X_train ,X_test,Y_train,Y_test= train_test_split(train_data,train_label,test_size=0.1)

print(np.array(Y_train).shape)



es = keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=3, baseline=0.99)

checkpoint = ModelCheckpoint('best_weights.h5', monitor='val_loss', sava_best_only=True, mode='auto', period=1)



#prova resnet

model = define_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#per il resnet..!!



#base_path = '/kaggle/input/Kannada-MNIST/'

#train_res = pd.read_csv(base_path + 'train.csv')

#test_res = pd.read_csv(base_path + 'test.csv')

#dig_mnist_res = pd.read_csv(base_path + 'Dig-MNIST.csv')



#X_train_images = []#

#x_list = []



#count = 0

#mappa = []

#len = train_res.shape[0]





#for i in range(0,len):

#    save_dir = '/kaggle/tmp/'

    

##    record = train_res.loc[i]

#    tmp_img_label = record['label']

 #   record.drop('label',inplace=True)

 #   record = np.array(record).reshape(28,28,1)

 #   if not os.path.exists(save_dir):

 #       os.makedirs(save_dir)

 #   tmpImg = np.dstack([record,record,record])

 #   new_im = PIL.Image.fromarray(np.array(tmpImg).astype('uint8'))

  #  img_path = save_dir + "img" + str(count) + ".png"

  #  new_im.save(img_path)

   # mappa.append([img_path, tmp_img_label])

    #count = count + 1

#df = pd.DataFrame(df,columns = ['filename','label'])

#df = pd.DataFrame(mappa)

#df.head()





#df.to_csv('/kaggle/tmp/train.csv',index=False)




#mappaDUE = pd.read_csv('/kaggle/tmp/train.csv', names =['filename','label'])

#df = pd.DataFrame(mappaDUE)

#df.head()

#df.to_csv('/kaggle/tmp/train.csv',index=False)



    

#for i in os.listdir('/kaggle/tmp/'):

    #X_train_images.append(new_img)

    #img = tensorflow.keras.preprocessing.image.load_img('/kaggle/tmp/' + i, target_size=(224, 224))

    #x = tensorflow.keras.preprocessing.image.img_to_array(img)

    #del img

    #x_list.append(x)

    



#datagen_res = ImageDataGenerator(featurewise_center=False,

#        rescale=1./255,# set input mean to 0 over the dataset

#        samplewise_center=False,  # set each sample mean to 0

#        featurewise_std_normalization=False,  # divide inputs by std of the dataset

#        samplewise_std_normalization=False,  # divide each input by its std

#        zca_whitening=False,  # apply ZCA whitening

##        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

#        zoom_range = 0.15, # Randomly zoom image  era 0.05 messo a 0.15

#       width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

 #       height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

 #       horizontal_flip=False,  # randomly flip images

 #       vertical_flip=False,

 #       validation_split=0.1)  # randomly flip images

#mappaDUE['label'] = mappaDUE['label'].astype(str)



#def create_flow(datagen, subset):

#    return datagen_res.flow_from_dataframe(

#        mappaDUE, #dataframe contenente il mapping file - label

#        directory='/kaggle/tmp/', # cartella contente le immagini puntate!!

#        x_col='filename', 

#        y_col=['label'],

#        class_mode='raw',

 #       target_size=(128, 128),

#        batch_size=128,

#        subset=subset

#    )



#train_gen = create_flow(datagen_res, 'training')

#val_gen = create_flow(datagen_res, 'validation')

print(np.array(X_train).shape)

print(np.array(Y_train).shape)

#history = model.fit_generator(datagen.flow(X_train,Y_train),

#                              #train_gen,

#                              epochs = EPOCHS, validation_data =(X_test,Y_test), #val_gen,

#                              verbose = 1, callbacks=[es,checkpoint])





datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.2, # Randomly zoom image 

        width_shift_range=0.20,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.20,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





history = model.fit_generator(datagen.flow(X_train,Y_train),

                              epochs = EPOCHS, validation_data = (X_test,Y_test),

                              verbose = 1, callbacks=[es,checkpoint])
model_prediction = define_model()

model_prediction.load_weights('best_weights.h5')


#len_test = np.array(test_res).shape[0]

#test_res.drop('id', axis=1, inplace=True)

#mappaT = []



#count = 0

#for i in range(0,len_test):

#    save_dir = '/kaggle/tmpTst/'

    

#    record = test_res.loc[i]

    #tmp_img_label = record['label']

    #record.drop('label',inplace=True)

#    record = np.array(record).reshape(28,28,1)

#    if not os.path.exists(save_dir):

#        os.makedirs(save_dir)

#    tmpImg = np.dstack([record,record,record])

#    new_im = PIL.Image.fromarray(np.array(tmpImg).astype('uint8'))

#    img_path = save_dir + "img" + str(count) + ".png"

#     new_im.save(img_path)

#    mappaT.append([img_path, ''])

#    count = count + 1

    

#df = pd.DataFrame(mappaT)

#df.head()

#df.to_csv('/kaggle/tmpTst/test.csv',index=False)

#mappaDUETST = pd.read_csv('/kaggle/tmpTst/test.csv', names =['filename','label'])

#df = pd.DataFrame(mappaDUETST)

#df.head()

#df.to_csv('/kaggle/tmpTst/test.csv',index=False)





#def create_test_gen(datagen):

#    return datagen.flow_from_dataframe(

#        mappaDUETST,

#        directory='/kaggle/tmpTst/',

#        x_col='filename',

#        class_mode=None,

#        target_size=(224, 224),

#        batch_size=128,

#        shuffle=False

#    )



#test_gen = create_test_gen(datagen_res)
#results = model_prediction.predict_generator(test_gen)



results = model_prediction.predict(test_data)

# select the indix with the maximum probability

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission_file = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

submission_file['label'] = results

#submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

print(results)

submission_file.to_csv("submission.csv",index=False)
acc = history.history['acc']



epochs_ = range(0, EPOCHS)



plt.plot(epochs_ , acc, label='accuracy')

plt.xlabel('epochs')

plt.ylabel('accuracy')



plt.title('accuracy vs epochs')

plt.legend()