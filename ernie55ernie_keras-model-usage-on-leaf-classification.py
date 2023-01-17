%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import time



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing import image

from keras.utils import np_utils

from keras.models import Sequential, load_model

from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D

from keras.layers.core import Flatten, Dense, Dropout, Activation, Merge

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD, RMSprop, Adam

from keras.layers.advanced_activations import ELU, LeakyReLU, ThresholdedReLU, SReLU



from keras.callbacks import ProgbarLogger, ModelCheckpoint



from PIL import Image



target_size = (256, 256)

grayscale = True



# Relative path for the train, test, and submission file

train_path = '../input/train.csv'

test_path = '../input/test.csv'

submission_path = '../input/sample_submission.csv'

submission_output = './submission.csv'



def load_image(id):

    img_path = '../input/images/%d.jpg' % (id, )

    img = image.load_img(img_path,

                         grayscale=grayscale)

    img.thumbnail(target_size)

    bg = Image.new('L', target_size, (0,))

    bg.paste(

        img, (int((target_size[0] - img.size[0]) / 2), int((target_size[1] - img.size[1]) / 2))

    )

    img_arr = image.img_to_array(bg)

    

    return img_arr
# Load training data

train_data = pd.read_csv(train_path)

# load the ids in the training data set

x_ids = train_data.iloc[:, 0]

x_images = list()

for i in x_ids:

    x_images.append(load_image(i))

x_images = np.array(x_images)

plt.imshow(x_images[0].squeeze())

plt.show()

print('Shape of images', x_images[0].shape)

# Ignore the first column (id) and the second column (species)

x_features = train_data.iloc[:, 2:].values

print('Number of features', x_features.shape[1])



# Convert the species to category type

y = train_data['species']

# Get the corresponding categories list for species

le = LabelEncoder()

le.fit(y)

y = le.transform(y)



nb_classes = len(le.classes_)

print('Number of classes', nb_classes)

print('Number of instances', len(y))



plt.hist(y, bins=nb_classes)

plt.title('Number of instances in each class')

plt.xlabel('Class id')

plt.ylabel('Number of instances')

plt.show()



# convert a class vectors (integers) to a binary class matrix

y = np_utils.to_categorical(y)



# Load testing data

test_data = pd.read_csv(test_path)

test_ids = test_data.iloc[:, 0]

test_images = list()

for i in test_ids:

    test_images.append(load_image(i))

test_images = np.array(test_images)



# Load submission file

submission_data = pd.read_csv(submission_path)
# The folds are made by preserving the percentage of samples for each class

sss = StratifiedShuffleSplit(10, 0.2, random_state=15)

for train_index, test_index in sss.split(x_images, y):

	x_train_images, x_test_images, x_train_features, x_test_features = x_images[train_index], x_images[test_index], x_features[train_index], x_features[test_index]

	y_train, y_test = y[train_index], y[test_index]

    

print('Shape of x train images', x_train_images.shape)

print('Shape of x train features', x_train_features.shape)

print('Shape of y train', y_train.shape)

print('Shape of x test images', x_test_images.shape)

print('Shape of x test features', x_test_features.shape)

print('Shape of y test', y_test.shape)
def construct_feature_model():

    print('Contructing the model')

    

    model1 = Sequential([

        Dense(nb_classes * 2, input_shape=x_train_features.shape[1:]),

        BatchNormalization(),

        Activation('tanh'),

        Dropout(0.25)

    ])

    

    model2 = Sequential([

        Convolution2D(32, 3, 3, border_mode='same',

                            input_shape=x_images.shape[1:]),

        Activation('tanh'),

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

        Dropout(0.5),

        Flatten(),

        Dense(nb_classes),

        Activation('tanh')

    ])

    

    model = Sequential([

        Merge([model1, model2], mode='concat', concat_axis=1),

        Dense(nb_classes * 2),

        Activation('tanh'),

        Dropout(0.5),

        Dense(nb_classes),

        Activation('softmax')

    ])

    

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    

    model.compile(optimizer=optimizer,

              loss='categorical_crossentropy',

              metrics=['accuracy'])

    

    print('Finish construction of the model')

    return model



model = construct_feature_model()
print('Start to fit')

s = time.time()

# Save the parameter for the best model

best_model_file = 'leaf.h5'

best_model_cb = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=0, save_best_only=True)



# Fitting parameters

batch_size = 32

nb_epoch = 8

verbose = 2

callbacks = [best_model_cb]

validation_split = 0.0

validation_data = ([x_test_features, x_test_images], y_test)

#validation_data = (x_test_features, y_test)

shuffle = True

class_weight = None

sample_weight = None

data_augmentation = False



if not data_augmentation:

    print('Not using data augmentation')

    history = model.fit([x_features, x_images], y,

    #history = model.fit(x_features, y,

              batch_size=batch_size,

              nb_epoch=nb_epoch, 

              verbose=verbose,

              callbacks=callbacks,

              validation_split=validation_split,

              validation_data=validation_data,

              shuffle=shuffle,

              class_weight=class_weight,

              sample_weight=sample_weight)

print('Finish fitting\nFitting time', time.time() - s)
plt.plot(history.history['val_acc'])

plt.xlabel('Number of epoch')

plt.ylabel('Validation accrucy')

plt.title('Validation accuracy vs number of epoch')

plt.show()

print('Maximum accuracy', max(history.history['val_acc']))
plt.plot(history.history['val_loss'], color='r')

plt.xlabel('Number of epoch')

plt.ylabel('Categorical cross entropy loss')

plt.title('Categorical cross entropy loss vs number of epoch')

plt.show()

print('Minimum loss', min(history.history['val_loss']))
model = load_model(best_model_file)



y_prob = model.predict([test_data.iloc[:, 1:].values, test_images]) # Remove id column

#y_prob = model.predict(test_data.iloc[:, 1:].values) # Remove id column



submission_data.iloc[:, 1:] = y_prob

submission_data.tail()



f = open(submission_output, 'w')

f.write(pd.DataFrame(submission_data).to_csv(index = False))

f.close()
'''

file_name = 'data.h5'

if os.path.isfile(file_name):

    os.remove(file_name)



h5f = h5py.File(file_name, 'w')

h5f.create_dataset('x_train_images', data=x_train_images)

h5f.create_dataset('x_train_features', data=x_train_features)

h5f.create_dataset('y_train', data=y_train)

h5f.create_dataset('x_test_images', data=x_test_images)

h5f.create_dataset('x_test_features', data=x_test_features)

h5f.create_dataset('y_test', data=y_test)



h5f.create_dataset('test_images', data=test_images)

h5f.create_dataset('test_faetures', data=test_data.iloc[:, 1:].values)



h5f.close()





    model = Sequential([

        Dense(nb_classes * 2, input_dim=x_train_features.shape[1]),

        BatchNormalization(),

        Activation('relu'),

        Dropout(0.5),

        Dense(nb_classes * 2),

        Activation('relu'),

        Dropout(0.5),

        Dense(nb_classes),

        Activation('softmax'),

    ])

'''