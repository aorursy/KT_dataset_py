!ls '../input'
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import metrics



from sklearn.utils import class_weight

from collections import Counter



import matplotlib.pyplot as plt



from os import listdir

from os.path import isfile, join



import pandas as pd
train_loc = '../input/crackling-sound/Data-Crackling'

#test_loc = '../input/testing/Test'
trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory=train_loc, target_size=(224,224))

#tsdata = ImageDataGenerator()

#testdata = tsdata.flow_from_directory(directory=test_loc, target_size=(224,224))
#diagnosis_csv = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'

#diagnosis = pd.read_csv(diagnosis_csv, names=['pId', 'diagnosis'])

#diagnosis.head()
categories = diagnosis['diagnosis'].unique()

categories
#vgg16 = VGG16(weights='imagenet')

#vgg16.summary()



#x  = vgg16.get_layer('fc2').output

#prediction = Dense(2, activation='softmax', name='predictions')(x)

IMAGE_SIZE = [224, 224]

vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# don't train existing weights

for layer in vgg.layers:

    layer.trainable = False

  



  

#  # useful for getting number of classes

#folders = glob('Datasets/Train/*')

  



# our layers - you can add more if you want

x = Flatten()(vgg.output)

# x = Dense(1000, activation='relu')(x)

prediction = Dense(2, activation='softmax')(x)



# create a model object

model = Model(inputs=vgg.input, outputs=prediction)

# create a model object

#model = Model(inputs=vgg.input, outputs=prediction)



# view the structure of the model

model.summary()
for layer in model.layers:

    layer.trainable = False



for layer in model.layers[-20:]:

    layer.trainable = True

    print("Layer '%s' is trainable" % layer.name)  
opt = Adam(lr=0.000001)

model.compile(optimizer=opt, loss=binary_crossentropy, 

              metrics=['accuracy', 'mae'])

model.summary()
checkpoint = ModelCheckpoint("vgg16_base_res.h5", monitor='val_accuracy', verbose=1, 

                             save_best_only=True, save_weights_only=False, mode='auto')

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
counter = Counter(traindata.classes)                       

max_val = float(max(counter.values()))   

class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

class_weights
hist = model.fit(traindata,

                 steps_per_epoch=traindata.samples//traindata.batch_size, 

                 #validation_data=testdata, 

                 class_weight=class_weights, 

                 #validation_steps=testdata.samples//testdata.batch_size, 

                 epochs=15)

#callbacks=([checkpoint,early])
plt.plot(hist.history['loss'], label='train')

#plt.plot(hist.history['val_loss'], label='val')

plt.title('VGG16: Loss (0.000001 = Adam LR)')

plt.legend();

plt.show()



plt.plot(hist.history['accuracy'], label='train')

#plt.plot(hist.history['val_accuracy'], label='val')

plt.title('VGG16: Accuracy (0.000001 = Adam LR)')

plt.legend();

plt.show()



plt.plot(hist.history['mae'], label='train')

#plt.plot(hist.history['val_mae'], label='val')

plt.title('VGG16: MAE and Validation MAE (0.000001 = Adam LR)')

plt.legend();

plt.show()
model.save('./crackling_model1.h5')
model.get_weights()