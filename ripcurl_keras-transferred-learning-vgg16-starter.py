import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

import matplotlib.image as mplimg

from matplotlib.pyplot import imshow

from keras import optimizers



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



from keras import layers

from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout

from keras.models import Model



import keras.backend as K

from keras.models import Sequential



import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
os.listdir("../input/")
train_df = pd.read_csv("../input/train.csv")

train_df.head()
def prepareImages(data, m, dataset):

    print("Preparing images")

    X_train = np.zeros((m, 100, 100, 3))

    count = 0

    

    for fig in data['Image']:

        #load images into images of size 100x100x3

        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        X_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return X_train
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
X = prepareImages(train_df, train_df.shape[0], "train")

X /= 255
y, label_encoder = prepare_labels(train_df['Id'])
y.shape
from keras import applications



# This will load the whole VGG16 network, including the top Dense layers.

# Note: by specifying the shape of top layers, input tensor shape is forced

# to be (224, 224, 3), therefore you can use it only on 224x224 images.

#vgg_model = applications.VGG16(weights='imagenet', include_top=True)



# If you are only interested in convolution filters. Note that by not

# specifying the shape of top layers, the input tensor shape is (None, None, 3),

# so you can use them for any size of images.

vgg_model = applications.VGG16(weights='imagenet', include_top=False)



# If you want to specify input tensor

from keras.layers import Input

input_tensor = Input(shape=(100, 100, 3))

vgg_model = applications.VGG16(weights='imagenet',

                               include_top=False,

                               input_tensor=input_tensor)



# To see the models' architecture and layer names, run the following

vgg_model.summary()
vgg_model = applications.VGG16(weights='imagenet',

                               include_top=False,

                               input_shape=(100, 100, 3))



# Creating dictionary that maps layer names to the layers

layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])



# Getting output tensor of the last VGG layer that we want to include

x = layer_dict['block5_pool'].output



# Stacking a new simple convolutional network on top of it    



x = Flatten()(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.2)(x)

x = Dense(y.shape[1], activation='softmax')(x)



# Creating new model. Please note that this is NOT a Sequential() model.

from keras.models import Model

model = Model(input=vgg_model.input, output=x)

model.summary()
model.layers[:-3]


# Make sure that the pre-trained bottom layers are not trainable

for layer in model.layers[:-4]:

    layer.trainable = False

    

from keras import optimizers

# Do not forget to compile it

adam=optimizers.Adam(lr=0.001)

model.compile(loss='categorical_crossentropy',

                     optimizer=adam,

                     metrics=['accuracy'])
history = model.fit(X, y, epochs=10, batch_size=512, verbose=1, validation_split=0.2)

#gc.collect()
plt.plot(history.history['acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.show()
adam=optimizers.Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy',

                     optimizer=adam,

                     metrics=['accuracy'])
#model.load_weights()
X.shape, y.shape
history = model.fit(X, y, epochs=10, batch_size=64, verbose=1, validation_split=0.2)
test = os.listdir("../input/test/")

print(len(test))
len(test)
col = ['Image']

test_df = pd.DataFrame(test, columns=col)

test_df['Id'] = ''
X = prepareImages(test_df, test_df.shape[0], 'test')

X /= 255
predictions = model.predict(np.array(X), verbose=1)
for i, pred in enumerate(predictions):

    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))


test_df.to_csv('submission.csv', index=False)
import os

#os.chdir(r'kaggle/working')

    

from IPython.display import FileLink

FileLink(r'submission.csv')
model.save_weights('model_vgg_trans.hdf5')

from IPython.display import FileLink

FileLink(r'model_vgg_trans.hdf5')