!pip install split_folders
import split_folders



from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import metrics



from sklearn.utils import class_weight

from collections import Counter



import matplotlib.pyplot as plt



import os

from os import listdir

from os.path import isfile, join
os.makedirs('output')

os.makedirs('output/train')

os.makedirs('output/val')
!ls ../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images
img_loc = '../input/diabetic-retinopathy-224x224-gaussian-filtered/gaussian_filtered_images/gaussian_filtered_images/'



split_folders.ratio(img_loc, output='output', seed=1, ratio=(0.8, 0.2))
!ls output
train_loc = 'output/train/'

test_loc = 'output/val/'
trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory=train_loc, target_size=(224,224))

tsdata = ImageDataGenerator()

testdata = tsdata.flow_from_directory(directory=test_loc, target_size=(224,224))
vgg16 = VGG16(weights='imagenet')

vgg16.summary()



x  = vgg16.get_layer('fc2').output

prediction = Dense(5, activation='softmax', name='predictions')(x)



model = Model(inputs=vgg16.input, outputs=prediction)
for layer in model.layers:

    layer.trainable = False



for layer in model.layers[-16:]:

    layer.trainable = True

    print("Layer '%s' is trainable" % layer.name)  
opt = Adam(lr=0.000001)

model.compile(optimizer=opt, loss=categorical_crossentropy, 

              metrics=['accuracy'])

model.summary()
checkpoint = ModelCheckpoint("vgg16_diabetes.h5", monitor='val_accuracy', verbose=1, 

                             save_best_only=True, save_weights_only=False, mode='auto')

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
counter = Counter(traindata.classes)                       

max_val = float(max(counter.values()))   

class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

class_weights
hist = model.fit(traindata, steps_per_epoch=traindata.samples//traindata.batch_size, validation_data=testdata, 

                 class_weight=class_weights, validation_steps=testdata.samples//testdata.batch_size, 

                 epochs=80,callbacks=[checkpoint,early])
plt.plot(hist.history['loss'], label='train')

plt.plot(hist.history['val_loss'], label='val')

plt.title('VGG16: Loss and Validation Loss (0.0001 = Adam LR)')

plt.legend();

plt.show()



plt.plot(hist.history['accuracy'], label='train')

plt.plot(hist.history['val_accuracy'], label='val')

plt.title('VGG16: Accuracy and Validation Accuracy (0.0001 = Adam LR)')

plt.legend();

plt.show()