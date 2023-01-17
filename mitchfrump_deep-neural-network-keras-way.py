import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



from keras.models import Sequential, load_model

from keras.layers import Dense , Dropout , Lambda, Flatten

from keras.optimizers import Adam ,RMSprop

from sklearn.model_selection import train_test_split

from keras import  backend as K

from keras.preprocessing.image import ImageDataGenerator



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# create the training & test sets, skipping the header row with [1:]

train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px



from keras.utils.np_utils import to_categorical

y_train= to_categorical(y_train)

num_classes = y_train.shape[1]
from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D

from keras.preprocessing import image

from sklearn.model_selection import train_test_split

X = X_train

y = y_train

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)



gen =ImageDataGenerator(rotation_range=12, width_shift_range=0.16, shear_range=0.6,

                               height_shift_range=0.16, zoom_range=0.16)

batches = gen.flow(X_train, y_train, batch_size=64)

val_batches = gen.flow(X_val, y_val, batch_size=64)



from keras.layers.normalization import BatchNormalization

from keras.layers.noise import GaussianDropout



def get_bn_model(size,dropout):

    model = Sequential([

        Convolution2D(8*size,(3,3), activation='relu', input_shape=(28,28,1)),

        GaussianDropout(dropout),

        BatchNormalization(),

        Convolution2D(8*size,(3,3), activation='relu'),

        MaxPooling2D(),

        GaussianDropout(dropout),

        BatchNormalization(),

        Convolution2D(16*size,(3,3), activation='relu'),

        GaussianDropout(dropout),

        BatchNormalization(),

        Convolution2D(16*size,(3,3), activation='relu'),

        MaxPooling2D(),

        Flatten(),

        GaussianDropout(dropout),

        BatchNormalization(),

        Dense(128*size, activation='relu'),

        GaussianDropout(dropout),

        BatchNormalization(),

        Dense(10, activation='softmax')

        ])

    model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def augment_and_create_model(model_size, dropout_rate, aug):

    model = get_bn_model(model_size,dropout_rate)

    gen = ImageDataGenerator(rotation_range=3*aug, width_shift_range=.04*aug, shear_range=.15*aug, height_shift_range=.04*aug, zoom_range=.04*aug)

    return model, gen.flow(X_train, y_train, batch_size = 64), gen.flow(X_val, y_val, batch_size=64)
models = []

epochs_to_train = 20

for x in range(4,5):

    aug = 3

    model_size = x

    dropout_rate = 4*.1

    print("Training a model with size level {}, dropout rate {}, and augmentation level {}".format(model_size,dropout_rate,aug))

    pretrained = 0

    print("Attempting to load saved model.")

    for pretrained_epochs in range(epochs_to_train, 0, -1):

        try:

            model = load_model('model_size_{}_dropout_{}_augment_{}_epochs_{}.h5'.format(x,.1*x,x,pretrained_epochs))

            print("Loaded existed model trained for {} epochs out of {}".format(pretrained_epochs,epochs_to_train))

            pretrained = pretrained_epochs

            model.compile(Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

            break;

        except:

            pass

    if pretrained == 0:

        print("Failed to load trained model. Creating new model.")

        this_model = get_bn_model(model_size,dropout_rate)

    else:

        this_model = model

    train_gen = ImageDataGenerator(rotation_range=3*aug, width_shift_range=.04*aug, shear_range=.15*aug, height_shift_range=.04*aug, zoom_range=.04*aug)

    batches = train_gen.flow(X_train, y_train, batch_size = 64)

    val_gen = ImageDataGenerator()

    val_batches = val_gen.flow(X_val, y_val, batch_size=64)

    this_model.optimizer.lr=0.01

    models.append(this_model)

    

    history=models[-1].fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=epochs_to_train-pretrained,

                        validation_data=val_batches, validation_steps=val_batches.n)

    this_model.save('model_size_{}_dropout_{}_augment_{}_epochs_{}.h5'.format(x,.1*x,x,epochs_to_train), include_optimizer=False)
for i in range(len(models)):

    model = models[i]

    plt.plot(model.history.history['val_loss'], label=i)

    print(i,model.history.history['val_loss'])

plt.legend()

plt.show()
best_loss = 0

best_loss_index = -1

for i in range(len(models)):

    model = models[i]

    print(model.history.history)

    this_loss = model.history.history['val_loss'][-1] + (model.history.history['val_loss'][-1] - model.history.history['loss'][-1])**2

    if this_loss > best_loss:

        best_loss = this_loss

        best_loss_index = i
predictions = models[i].predict_classes(X_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)