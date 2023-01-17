
%reset -f

## 1. Call libraries
import numpy as np

# 1.1 Classes for creating models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense

# 1.2 Class for accessing pre-built models
from tensorflow.keras import applications

# 1.3 Class for generating infinite images
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1.4 Miscelleneous
import matplotlib.pyplot as plt
import time, os
train_data_dir = '..//input//intel-image-classification//seg_train//seg_train'

validation_data_dir = '..//input//intel-image-classification//seg_test//seg_test'

img_width, img_height = 75,75
nb_train_samples, nb_validation_samples = 14000, 3000
batch_size = 25
bf_filename = '..//working//bottleneck_features_train.npy'
val_filename = '..//working//bottleneck_features_validation.npy'

datagen_train = ImageDataGenerator(rescale=1. / 255)

generator_tr = datagen_train.flow_from_directory(
              directory = train_data_dir,     
              target_size=(img_width, img_height),    
              batch_size=batch_size,                  
              class_mode=None,                        
              shuffle=False                           
                                                    
              )
datagen_val = ImageDataGenerator(rescale=1. / 255)
generator_val = datagen_val.flow_from_directory(
                                          validation_data_dir,
                                          target_size=(img_width, img_height),
                                          batch_size=batch_size,
                                          class_mode=None,
                                          shuffle=False   
                                                   
                                          )
#Buld VGG16 network model with 'imagenet' weights

model = applications.VGG16(
	                       include_top=False,
	                       weights='imagenet',
	                       input_shape=(img_width, img_height,3)
	                       )

model.summary()
start = time.time()


bottleneck_features_train = model.predict_generator(
                                                    generator = generator_tr,
                                                    steps = nb_train_samples // batch_size,
                                                    verbose = 1
                                                    )
end = time.time()

print("Time taken: ",(end - start)/60, "minutes")
start = time.time()
bottleneck_features_validation = model.predict_generator(
                                                         generator = generator_val,
                                                         steps = nb_validation_samples // batch_size,
                                                         verbose = 1
                                                         )

end = time.time()

print("Time taken: ",(end - start)/60, "minutes")
if os.path.exists(bf_filename):
    os.system('rm ' + bf_filename)


np.save(open(bf_filename, 'wb'), bottleneck_features_train)


if os.path.exists(val_filename):
    os.system('rm ' + val_filename)


np.save(open(val_filename, 'wb'), bottleneck_features_validation)
%reset -f

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Softmax
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras import applications
import matplotlib.pyplot as plt
import time, os
img_width, img_height = 75,75  

nb_train_samples = 14000


nb_validation_samples = 3000

epochs = 50

batch_size = 64

num_classes = 6
bf_filename = '..//working//bottleneck_features_train.npy'
val_filename = '..//working//bottleneck_features_validation.npy'

top_model_weights_path = '..//working///bottleneck_fc_model.h5'

train_data_features = np.load(open(bf_filename,'rb'))

train_data_features.shape 
train_labels = np.array([1] * 2300 + [2] * 2300 + [3] * 2300 + [4] * 2300 + [5] * 2300 + [6] * 2300)   

train_labels

train_labels.shape 
x = np.arange(13800)      

np.random.shuffle(x)     

x
train_data_features = train_data_features[x, :,:,:]

train_labels = train_labels[x]

train_labels.shape

train_labels_categ = to_categorical(train_labels , )

train_labels_categ.shape

train_labels_categ         
train_labels_categ = train_labels_categ[:, 1:]

train_labels_categ
validation_data_features = np.load(open(val_filename,'rb')) 

validation_data_features.shape
  
validation_labels = np.array([1] * 500 + [2] * 500 + [3] * 500 + [4] * 500 + [5] * 500 + [6] * 500)

validation_labels = to_categorical(validation_labels)

validation_labels = validation_labels[:,1:]

validation_labels.shape
validation_labels.shape[1:]
model = Sequential()

model.add(Flatten(input_shape=train_data_features.shape[1:]))     

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(
              optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )
# 5.2 Fit model and make predictions on validation dataset
#     Takes 2 minutes
#     Watch Validation loss and Validation accuracy (around 81%)
start = time.time()
history = model.fit(train_data_features, train_labels_categ,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data_features, validation_labels),
                    verbose =1
                   )
end = time.time()

print("Time taken: ",(end - start)/60, "minutes")

history.history.keys()

len(history.history['accuracy'])

len(history.history['val_accuracy'])


def plot_learning_curve():
    val_acc = history.history['val_accuracy']
    tr_acc=history.history['accuracy']
    epochs = range(1, len(val_acc) +1)
    plt.plot(epochs,val_acc, 'b', label = "Validation accu")
    plt.plot(epochs, tr_acc, 'r', label = "Training accu")
    plt.title("Training and validation accuracy")
    plt.xlabel("epochs-->")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
plot_learning_curve()






