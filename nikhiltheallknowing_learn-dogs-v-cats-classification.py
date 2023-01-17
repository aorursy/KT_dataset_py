import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from time import sleep

from random import shuffle

from tqdm import tqdm

from skimage.io import imread

from skimage.transform import resize

from matplotlib import pyplot as plt

from IPython.display import clear_output



# DeepLearning with Keras libraries!

from keras.callbacks import Callback

from keras.models import Sequential, Model

from keras.layers import Input ,Dense, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Flatten, Reshape, concatenate
root = '../input/dog vs cat/dataset/'

LABELS = ['CAT', 'DOG']
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []

        



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('Log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="acc")

        ax2.plot(self.x, self.val_acc, label="val_acc")

        ax2.legend()

        

        plt.show()

        

        

plot = PlotLearning()
training_set = []

test_set = []



#print('Loading Training Data...')

for animal in os.listdir(root+'training_set/'):

    for file in tqdm(os.listdir(root+'training_set/'+animal)):

        img_data = imread(root+'training_set/'+animal+'/'+file)

        img_data = resize(img_data, output_shape=(128,128))

        training_set.append([img_data, [1, 0] if animal == 'cats' else [0,1]])

        

#print('Loading Testing Data...')

for animal in os.listdir(root+'test_set/'):

    for file in tqdm(os.listdir(root+'test_set/'+animal)):

        img_data = imread(root+'test_set/'+animal+'/'+file)

        img_data = resize(img_data, output_shape=(128,128))

        test_set.append([img_data, [1, 0] if animal == 'cats' else [0,1]])

        

plt.hist(np.argmax(Y, axis=1))

plt.title("0-Cat & 1-Dog")

plt.show()
shuffle(training_set)

shuffle(test_set)

X, X_ = [], []

Y, Y_ = [], []



for i in tqdm(training_set):

    X.append(i[0])

    Y.append(i[1])



for i in tqdm(test_set):

    X_.append(i[0])

    Y_.append(i[1])



del training_set

del test_set



X, Y, X_, Y_ = np.array(X)/255, np.array(Y), np.array(X_)/255, np.array(Y_)
input_layer = Input(shape=(128, 128, 3))



x1 = Conv2D(16, (3,3), activation='relu')(input_layer)

x2 = Conv2D(16, (3,3), activation='relu')(input_layer)



x1 = MaxPooling2D((3,3))(x1)

x2 = MaxPooling2D((4,4))(x2)



x1 = BatchNormalization()(x1)

x2 = BatchNormalization()(x2)



x1 = Conv2D(32, (3,3), activation='relu')(x1)

x2 = Conv2D(32, (3,3), activation='relu')(x2)



x1 = MaxPooling2D((3,3))(x1)

x2 = MaxPooling2D((4,4))(x2)



x1 = BatchNormalization()(x1)

x2 = BatchNormalization()(x2)



x1 = Flatten()(x1)

x2 = Flatten()(x2)



x = concatenate([x1, x2]) # All Branches JOined to `x` node



x = Dense(1024, activation='relu')(x)



output_layer = Dense(2, activation='softmax', name='output_layer')(x)



model = Model(inputs=input_layer, outputs=output_layer)





#model.build(input_shape=(None ,128 ,128 ,3))

model.compile(

    optimizer='Adadelta',

    loss='binary_crossentropy',

    metrics=['accuracy']

)



model.summary()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.fit(X, Y, epochs=20, validation_data=(X,Y), batch_size=100, callbacks=[plot])
model.save('my_model.h5')