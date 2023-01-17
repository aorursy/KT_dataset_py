# files loading

import os

import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
# image pretreatment

import matplotlib.pyplot as plt



import cv2

import numpy as np

from PIL import Image

from keras.utils.vis_utils import plot_model
import keras

from numpy import expand_dims

from keras import backend as K

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import ImageDataGenerator

import imageio

import imgaug as ia

from imgaug import augmenters as iaa
# libraries for a CNN

import tensorflow as tf



from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

from keras.layers.normalization import BatchNormalization

from keras.layers import GlobalAveragePooling2D

from keras.layers import Dropout, Activation

from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop

from keras.utils import to_categorical



from tensorflow.python.client import device_lib

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
ANNOTATION_DIR = '../input/stanford-dogs-dataset/annotations/Annotation' 

IMAGES_DIR = '../input/stanford-dogs-dataset/images/Images'
breed_list = os.listdir(IMAGES_DIR)

print("num. breeds total:", len(breed_list))
filtered_breeds = [breed.split('-',1)[1] for breed in breed_list] #visualize breeds

filtered_breeds[:12]
def show_dir_images(breed, n_to_show):

    plt.figure(figsize=(16,16))

    img_dir = "../input/stanford-dogs-dataset/images/Images/{}/".format(breed)

    images = os.listdir(img_dir)[:n_to_show]

    for i in range(n_to_show):

        img = cv2.imread(img_dir + images[i])

        plt.subplot(n_to_show/4+1, 4, i+1)

        plt.imshow(img)

        plt.axis('off')







print(breed_list[11])

show_dir_images(breed_list[11], 4)
img_dir = "../input/stanford-dogs-dataset/images/Images/{}/".format(breed_list[11])

images = os.listdir(img_dir)[:12]

images = os.listdir(img_dir)[:4]

img = cv2.imread(img_dir + images[0])



# transform image for equalization

img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)



plt.imshow(img_to_yuv)
hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')

plt.hist(img_to_yuv.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])

equ = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)

res = np.hstack((img_to_yuv,equ)) #stacking images side-by-side

cv2.imwrite('res.png',res)

plt.imshow(res)
hist,bins = np.histogram(equ.flatten(),256,[0,256])

cdf = hist.cumsum()

cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')

plt.hist(equ.flatten(),256,[0,256], color = 'r')

plt.show()
print("Augmented by rotation:")

#ia.imshow(image_aug_2)

# convert to numpy array

data = img_to_array(img_RGB)



# expand dimension to one sample

samples = expand_dims(data, 0)

# create image data augmentation generator

datagen = ImageDataGenerator(rotation_range=30)

# prepare iterator

it = datagen.flow(samples, batch_size=1)

# generate samples and plot

for i in range(9):

    # define subplot

    plt.subplot(330 + 1 + i)

    # generate batch of images

    batch = it.next()

    # convert to unsigned integers for viewing

    image = batch[0].astype('uint8')

    # plot raw pixel data

    plt.imshow(image)

# show the figure

plt.figure(figsize = (15,15))

plt.show()
num_breeds = 10 # integer between 2 and 120

breeds = breed_list[:num_breeds]



def load_images_and_labels(breeds):

    img_lst=[]

    labels=[]

    

    for index, breed in enumerate(breeds):

        for image_name in os.listdir(IMAGES_DIR+"/"+breed):

            img = cv2.imread(IMAGES_DIR+"/"+breed+"/"+image_name)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            

            img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0]) # convert to yuv color space for equalization

            equ = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB) # equalize

            

            img_array = Image.fromarray(img, 'RGB')

            

            #resize image to 227 x 227 because the input image resolution for AlexNet is 227 x 227

            resized_img = img_array.resize((227, 227))

            

            img_lst.append(np.array(resized_img))

            

            labels.append(filtered_breeds[index])

            

    return img_lst, labels 



images, labels = load_images_and_labels(breeds)

print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
# replace numbers with names

le = LabelEncoder()

nlabels = le.fit_transform(labels) # encode labels as number values. This prepares for categorical encoding

Y=to_categorical(nlabels,num_classes = num_breeds) # category encoding
#Normalization for the images

images = np.array(images)

images = images.astype(np.float32)

#labels = labels.astype(np.int32)

X_norm = images/255
x_train, x_test, y_train, y_test = train_test_split(X_norm, Y, test_size = 0.2, random_state = 42)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=1)



print("x_train shape = ",x_train.shape)

print("y_train shape = ",y_train.shape)



print("\nx_val shape = ",x_val.shape)

print("y_val shape = ",y_val.shape)



print("\nx_test shape = ",x_test.shape)

print("y_test shape = ",y_test.shape)
# example of training images

df_y_train = pd.DataFrame(y_train, columns = filtered_breeds[:num_breeds]) 

df_given_train = df_y_train.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=1)

plt.figure(figsize = (20,20))

for i in range(5):

    img = x_train[i]

    plt.subplot(1,5,i+1)

    plt.imshow(img)

    #plt.axis("off")

    plt.xlabel(y_train[i], color = "r")

    plt.title(df_given_train.iloc[i,0])

plt.show()
aug = ImageDataGenerator(rotation_range=30, #rotations (as seen above)

                        width_shift_range=0.2,  # randomly shift images horizontally 

                        height_shift_range=0.2,# randomly shift images vertically 

                        shear_range=0.2, # shear image

                        zoom_range=0.2, # zoom into image 

                        horizontal_flip=True, # randomly flip images horizontally

                        fill_mode='reflect') #  creates a ‘reflection’ and fills the empty values in reverse order of the known values

# fit parameters from data

aug.fit(x_train, augment=True)
input_shape = (None, 227, 227, 3)

 



model1 = tf.keras.Sequential() 

model1.add(tf.keras.layers.Conv2D(16, (3, 3), use_bias=False))

model1.add(tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same'))        

model1.add(tf.keras.layers.Flatten())        

model1.add( tf.keras.layers.Dense(512, activation='relu'))       

model1.add(tf.keras.layers.Dense(num_breeds, activation='softmax'))        



model1.build(input_shape)

model1.summary()

model1.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, amsgrad=False), metrics=['accuracy']) #compile model



plot_model(model1, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
%%time

history_1 = model1.fit(aug.flow(x_train, y_train), 

          validation_data=(x_val, y_val),

          epochs=5)
plt.figure(figsize=(18, 6))



plt.subplot(121)

loss = history_1.history['loss']

val_loss = history_1.history['val_loss']

plt.plot(loss,"--", linewidth=3 , label="train")

plt.plot(val_loss, linewidth=3 , label="valid")



plt.legend(['train','validation'], loc='upper left')

plt.grid()

plt.ylabel('loss')

plt.ylim((1.5, 3))

plt.xlabel('Epoch')

plt.title('2 layers CNN Model Loss')

plt.legend(['train','validation'], loc='upper left')



plt.subplot(122)

acc = history_1.history['accuracy']

val_acc = history_1.history['val_accuracy']



plt.plot(acc,"--", linewidth=3 , label="train")

plt.plot(val_acc, linewidth=3 , label="valid")



plt.legend(['train','validation'], loc='upper left')

plt.grid()



plt.ylabel('accuracy')

plt.xlabel('Epoch')

plt.title('2 layers CNN Model accuracy')

plt.legend(['train','validation'], loc='upper left')

plt.show()
test_loss, test_accuracy = model1.evaluate(x_test, y_test)



print("Test results \n Loss:",test_loss,'\n Accuracy',test_accuracy)
# Initialize the classifier.

classifier = Sequential()

          

# 1st Convolutional Layer

classifier.add(Conv2D(filters = 96, input_shape = (227,227,3), kernel_size = (11,11), strides = (4,4), padding = 'valid'))

classifier.add(Activation('relu'))

# Batch Normalisation before passing it to the next layer

classifier.add(BatchNormalization())

# Pooling Layer

classifier.add(MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'valid'))



# 2nd Convolutional Layer

classifier.add(Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), padding = 'same'))

classifier.add(Activation('relu'))

# Batch Normalisation

classifier.add(BatchNormalization())

# Pooling Layer

classifier.add(MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'valid'))



# 3rd Convolutional Layer

classifier.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'same'))

classifier.add(Activation('relu'))

# Batch Normalisation

classifier.add(BatchNormalization())

# Dropout

classifier.add(Dropout(0.5))



# 4th Convolutional Layer

classifier.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'same'))

classifier.add(Activation('relu'))

# Batch Normalisation

classifier.add(BatchNormalization())

# Dropout

classifier.add(Dropout(0.5))



# 5th Convolutional Layer

classifier.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same'))

classifier.add(Activation('relu'))

# Batch Normalisation

classifier.add(BatchNormalization())

# Pooling Layer

classifier.add(MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'valid'))

# Dropout

classifier.add(Dropout(0.5))



# Passing it to a dense layer

classifier.add(Flatten())



# 1st Dense Layer

classifier.add(Dense(4096, input_shape = (227,227,3)))

classifier.add(Activation('relu'))

# Add Dropout to prevent overfitting

classifier.add(Dropout(0.5))

# Batch Normalisation

classifier.add(BatchNormalization())



# 2nd Dense Layer

classifier.add(Dense(4096))

classifier.add(Activation('relu'))

# Add Dropout

classifier.add(Dropout(0.3))

# Batch Normalisation

classifier.add(BatchNormalization())



# 3rd Dense Layer

classifier.add(Dense(1000))

classifier.add(Activation('relu'))

# Add Dropout

classifier.add(Dropout(0.2))

# Batch Normalisation

classifier.add(BatchNormalization())



# Output Layer

classifier.add(Dense(num_breeds))

classifier.add(Activation('softmax'))



# Get the classifier summary.

classifier.summary()



classifier.build(input_shape)
#classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Compile 

INIT_LR = 1e-3

DECAY = 1e-7



opt = tf.keras.optimizers.Adam(lr = INIT_LR, decay = DECAY)

classifier.compile(loss="categorical_crossentropy", optimizer = opt,metrics = ["accuracy"])

print("[INFO] Training network...")



#¤K.set_value(classifier.optimizer.learning_rate, 0.00001)
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
%%time

history_2 = classifier.fit( aug.flow(x_train, y_train,  shuffle = False),

    validation_data = (x_val, y_val),

    epochs = 40,

    verbose=1)



#hist = model.fit(datagen.flow(x_train, y_train), epochs=25)
## plot the history of loss and accuracy for train and valid data for the Alexnet model

plt.figure(figsize=(18, 6))



plt.subplot(121)

loss = history_2.history['loss']

val_loss = history_2.history['val_loss']

plt.plot(loss,"--", linewidth=3 , label="train")

plt.plot(val_loss, linewidth=3 , label="valid")



plt.legend(['train','validation'], loc='upper left')

plt.grid()

plt.ylabel('loss')

plt.ylim((1, 6))

plt.xlabel('Epoch')

plt.title('Alexnet CNN Model Loss')

plt.legend(['train','validation'], loc='upper left')



plt.subplot(122)

acc = history_2.history['accuracy']

val_acc = history_2.history['val_accuracy']



plt.plot(acc,"--", linewidth=3 , label="train")

plt.plot(val_acc, linewidth=3 , label="valid")



plt.legend(['train','validation'], loc='upper left')

plt.grid()



plt.ylabel('accuracy')

plt.xlabel('Epoch')

plt.title('Alexnet CNN Model accuracy')

plt.legend(['train','validation'], loc='upper left')

plt.show()
test_loss, test_accuracy = classifier.evaluate(x_test, y_test)



print(test_loss,test_accuracy)
pred = classifier.predict(x_test)

print(y_test.shape)

print(pred.shape)
roundpred = np.around(pred, decimals=1)

df_pred = pd.DataFrame(roundpred, columns = filtered_breeds[:num_breeds])

df_pred.head()
df_breed_pred = df_pred.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=3)

df_breed_pred.columns = ['1st_prob_breed','2nd_prob_breed','3rd_prob_breed']

df_breed_pred.head()
prob_df = df_pred.apply(np.sort, axis=1).apply(lambda df_pred: df_pred[-3:]).apply(pd.Series)

prob_df.columns = ['3rd_prob','2nd_prob','1st_prob']

prob_df = prob_df*100

prob_df = prob_df.astype(int)

prob_df = pd.concat([prob_df, df_breed_pred], axis=1)
prob_df['final']= prob_df["1st_prob_breed"].astype(str) +" "+ prob_df["1st_prob"].astype(str)+"%, "+prob_df["2nd_prob_breed"].astype(str) +" "+ prob_df["2nd_prob"].astype(str)+"%, "+prob_df["3rd_prob_breed"].astype(str) +" "+ prob_df["3rd_prob"].astype(str)+"%"
df_test = pd.DataFrame(y_test, columns = filtered_breeds[:num_breeds])

given_df = df_test.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=1)

given_df.head()

plt.figure(1 , figsize = (19 , 10))

n = 0 

r = np.random.randint(low=1, high=100, size=9)

for i in r:

    n += 1 

    

    

    plt.subplot(3, 3, n)

    plt.subplots_adjust(hspace = 0.3, wspace = 0.3)

    

    plt.imshow(x_test[i])

    plt.title(given_df.iloc[i,0])

 

    plt.xlabel(prob_df.iloc[i,6], wrap=True, color = "r")

    plt.xticks([]) , plt.yticks([])



plt.show()