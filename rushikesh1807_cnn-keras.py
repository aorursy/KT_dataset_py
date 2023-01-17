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



root=("../input/bizlerstechnologieshiringchallenge2019/BizlersTechnologiesHiringChallenge2019/")

LABELS=['A','B','C','D']

Test_LABELS=['t1','t2','t3','t4']
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
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(root+'Train/',

                                                 target_size = (128, 128),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')



test_set = test_datagen.flow_from_directory(root+'Test/',

                                            target_size = (128, 128),

                                            batch_size = 32,

                                            class_mode = 'categorical')
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



x = Dense(512, activation='relu')(x)



output_layer = Dense(4, activation='softmax', name='output_layer')(x)



model = Model(inputs=input_layer, outputs=output_layer)





#model.build(input_shape=(None ,128 ,128 ,3))

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)



model.summary()
from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot



SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.fit_generator(training_set,epochs=25,steps_per_epoch = training_set.samples // 32,

    validation_data = test_set, 

    validation_steps = test_set.samples // 32, callbacks=[plot,EarlyStopping(patience=3, restore_best_weights=True),

        ReduceLROnPlateau(patience=2)])

y_pred=[]

y_test=[]

from keras.preprocessing import image

for charactor in os.listdir(root+'Test/'):

    for file in tqdm(os.listdir(root+'Test/'+charactor)):

        img_data = imread(root+'Test/'+charactor+'/'+file)

        img_data = resize(img_data, output_shape=(128,128))

        img_data = np.expand_dims(img_data, axis = 0)

        y_test.append(Test_LABELS.index(charactor))

        y_pred.append(np.argmax(model.predict(img_data)))

        print(Test_LABELS.index(charactor)==(np.argmax(model.predict(img_data))))

        
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")