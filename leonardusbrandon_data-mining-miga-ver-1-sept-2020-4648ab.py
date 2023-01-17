# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# TRANSFER LEARNING (RESNET50)



from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from sklearn.metrics import f1_score

import keras



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 2

weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['Precision', 

                        'Recall','AUC'])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/train-gemastik-september/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/test-gemastik-september/test',

        target_size =(image_size, image_size),

        class_mode ='categorical')



data_history_Res = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=125,

        validation_data=validation_generator,

        validation_steps=1)
# Plotting Training and Validation loss - Transfer Learning ResNet50



loss_train_Res = data_history_Res.history['loss']

loss_val_Res = data_history_Res.history['val_loss']

epochs = range(125)



plt.plot(epochs,loss_train_Res,'g',label='Training loss ResNet50')

plt.plot(epochs,loss_val_Res,'b',label='Validation loss ResNet50')

plt.title('Training and Validation loss - Transfer Learning ResNet50')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
# Plotting Val Precision, Recall, AUC Score - Transfer Learning ResNet50



val_precision_Res = data_history_Res.history['val_precision']

val_recall_Res = data_history_Res.history['val_recall']

val_auc_Res = data_history_Res.history['val_auc']

epochs = range(125)



plt.plot(epochs, val_precision_Res, 'c', label='Validation Precision ResNet50')

plt.plot(epochs, val_recall_Res, 'm', label='Validation Recall ResNet50')

plt.plot(epochs, val_auc_Res, 'g', label='Validation AUC ResNet50')

plt.title('Precision, Recall, and AUC Score - Transfer Learning ResNet50')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
# TRANSFER LEARNING (INCEPTIONV3)



from keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D 

from keras.callbacks import History 

history = History()



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 2

weights_path = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights=weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['Precision', 

                        'Recall','AUC'])



# Fitting Model

from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 299

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/train-gemastik-september/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/test-gemastik-september/test',

        target_size=(image_size, image_size),

        class_mode='categorical')



data_history_Inc = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=125,

        validation_data=validation_generator,

        validation_steps=1)
# Plotting Training and Validation loss - Transfer Learning InceptionV3



loss_train_Inc=data_history_Inc.history['loss']

loss_val_Inc=data_history_Inc.history['val_loss']

epochs = range(125)



plt.plot(epochs,loss_train_Inc,'g',label='Training Loss InceptionV3')

plt.plot(epochs,loss_val_Inc,'b',label='Validation Loss InceptionV3')

plt.title('Training and Validition loss - Transfer Learning InceptionV3')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
# Plotting Val Precision and Recall - Transfer Learning InceptionV3



val_precision_Inc = data_history_Inc.history['val_precision']

val_recall_Inc = data_history_Inc.history['val_recall']

val_auc_Inc = data_history_Inc.history['val_auc']

epochs = range(125)



plt.plot(epochs, val_precision_Inc, 'c', label='Validation Precision InceptionV3')

plt.plot(epochs, val_recall_Inc, 'm', label='Validation Recall InceptionV3')

plt.plot(epochs, val_auc_Inc, 'g', label='Validation AUC InceptionV3')

plt.title('Precision, Recall, and AUC Score - Transfer Learning InceptionV3')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
# TRANSFER LEARNING (VGG16)



from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from sklearn.metrics import f1_score

import keras



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 2

weights_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(VGG16(include_top=False, pooling='avg', weights=weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['Precision', 

                        'Recall','AUC'])



# Fitting Model

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)





train_generator = data_generator.flow_from_directory(

        '../input/train-gemastik-september/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



validation_generator = data_generator.flow_from_directory(

        '../input/test-gemastik-september/test',

        target_size =(image_size, image_size),

        class_mode ='categorical')



data_history_Vgg = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=125,

        validation_data=validation_generator,

        validation_steps=1)
# Plotting Training and Validation loss - Transfer Learning VGG16



loss_train_Vgg = data_history_Vgg.history['loss']

loss_val_Vgg = data_history_Vgg.history['val_loss']

epochs = range(125)



plt.plot(epochs,loss_train_Vgg,'g',label='Training Loss VGG16')

plt.plot(epochs,loss_val_Vgg,'b',label='Validation Loss VGG16')

plt.title('Training and Validition loss - Transfer Learning VGG16')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
# Plotting Val Precision and Recall - Transfer Learning VGG16



val_precision_Vgg = data_history_Vgg.history['val_precision']

val_recall_Vgg = data_history_Vgg.history['val_recall']

val_auc_Vgg = data_history_Vgg.history['val_auc']

epochs = range(125)



plt.plot(epochs, val_precision_Vgg, 'c', label='Validation Precision  VGG16')

plt.plot(epochs, val_recall_Vgg, 'm', label='Validation Recall VGG16')

plt.plot(epochs, val_auc_Vgg, 'g', label='Validation AUC VGG16')

plt.title('Precision, Recall, and AUC Score - Transfer Learning VGG16')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
# TRANSFER LEARNING WITH AUGMENTATION (ResNet50)



from tensorflow.keras.applications import ResNet50

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from keras.callbacks import History 

history = History()



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 2

weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['Precision', 

                        'Recall','AUC'])



# Fitting Model

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224



data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2)



train_generator = data_generator_with_aug.flow_from_directory(

        '../input/train-gemastik-september/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = data_generator_no_aug.flow_from_directory(

        '../input/test-gemastik-september/test',

        target_size=(image_size, image_size),

        class_mode='categorical')







data_history_Aug_Res = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=125,

        validation_data=validation_generator,

        validation_steps=1,callbacks=[history])
# Plotting Training and Validation loss - Augmentation ResNet50



loss_train_Aug_Res = data_history_Aug_Res.history['loss']

loss_val_Aug_Res = data_history_Aug_Res.history['val_loss']

epochs = range(125)



plt.plot(epochs,loss_train_Aug_Res,'g',label='Training loss Augmentation ResNet50')

plt.plot(epochs,loss_val_Aug_Res,'b',label='Validation loss Augmentation ResNet50')

plt.title('Training and Validation loss - Augmentation ResNet50')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
# Plotting Val Precision and Recall - Augmentation ResNet50



val_precision_Aug_Res = data_history_Aug_Res.history['val_precision']

val_recall_Aug_Res = data_history_Aug_Res.history['val_recall']

val_auc_Aug_Res = data_history_Aug_Res.history['val_auc']

epochs = range(125)



plt.plot(epochs, val_precision_Aug_Res, 'c', label='Validation Precision Augmentation ResNet50')

plt.plot(epochs, val_recall_Aug_Res, 'm', label='Validation Recall Augmentation ResNet50')

plt.plot(epochs, val_auc_Aug_Res, 'g', label='Validation AUC Augmentation ResNet50')

plt.title('Precision, Recall, and AUC Score - Augmentation ResNet50')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
# TRANSFER LEARNING WITH AUGMENTATION (InceptionV3)



from keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from keras.callbacks import History 

history = History()



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 2

weights_path = '../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(InceptionV3(include_top=False, pooling='avg', weights=weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (ResNet) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['Precision', 

                        'Recall','AUC'])



# Fitting Model

from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 299



data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2)



train_generator = data_generator_with_aug.flow_from_directory(

        '../input/train-gemastik-september/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = data_generator_no_aug.flow_from_directory(

        '../input/test-gemastik-september/test',

        target_size=(image_size, image_size),

        class_mode='categorical')







data_history_Aug_Inc = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=125,

        validation_data=validation_generator,

        validation_steps=1,callbacks=[history])
# Plotting Training and Validation loss - Augmentation InceptionV3



loss_train_Aug_Inc = data_history_Aug_Inc.history['loss']

loss_val_Aug_Inc = data_history_Aug_Inc.history['val_loss']

epochs = range(125)



plt.plot(epochs,loss_train_Aug_Inc,'g',label='Training loss Augmentation InceptionV3')

plt.plot(epochs,loss_val_Aug_Inc,'b',label='Validation loss Augmentation InceptionV3')

plt.title('Training and Validation loss - Augmentation InceptionV3')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
# Plotting Val Precision and Recall - Augmentation InceptionV3



val_precision_Aug_Inc = data_history_Aug_Inc.history['val_precision']

val_recall_Aug_Inc = data_history_Aug_Inc.history['val_recall']

val_auc_Aug_Inc = data_history_Aug_Inc.history['val_auc']

epochs = range(125)



plt.plot(epochs, val_precision_Aug_Inc, 'c', label='Validation Precision Augmentation InceptionV3')

plt.plot(epochs, val_recall_Aug_Inc, 'm', label='Validation Recall Augmentation InceptionV3')

plt.plot(epochs, val_auc_Aug_Inc, 'g', label='Validation AUC Augmentation InceptionV3')

plt.title('Precision, Recall, and AUC Score - Augmentation InceptionV3')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()
# TRANSFER LEARNING WITH AUGMENTATION (VGG16)



from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

from keras.callbacks import History 

history = History()



# So that the result is not randomized

seed_value=0



os.environ['PYTHONHASHSEED']=str(seed_value)



import random

random.seed(seed_value)



np.random.seed(seed_value)



import tensorflow as tf 

tf.compat.v1.set_random_seed(seed_value)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

tf.compat.v1.keras.backend.set_session(sess)



num_classes = 2

resnet_weights_path = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'



my_new_model = Sequential()

my_new_model.add(VGG16(include_top=False, pooling='avg', weights=resnet_weights_path))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Say not to train first layer (VGG16) model. It is already trained

my_new_model.layers[0].trainable = False



# Compile Model

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['Precision', 

                        'Recall','AUC'])



# Fitting Model

from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224



data_generator_with_aug = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2)



train_generator = data_generator_with_aug.flow_from_directory(

        '../input/train-gemastik-september/train',

        target_size=(image_size, image_size),

        batch_size=24,

        class_mode='categorical')



data_generator_no_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

validation_generator = data_generator_no_aug.flow_from_directory(

        '../input/test-gemastik-september/test',

        target_size=(image_size, image_size),

        class_mode='categorical')



data_history_Aug_Vgg = my_new_model.fit_generator(

        train_generator,

        steps_per_epoch=3,

        epochs=125,

        validation_data=validation_generator,

        validation_steps=1,callbacks=[history])
# Plotting Training and Validation loss - Augmentation VGG16



loss_train_Aug_Vgg = data_history_Aug_Vgg.history['loss']

loss_val_Aug_Vgg = data_history_Aug_Vgg.history['val_loss']

epochs = range(125)



plt.plot(epochs,loss_train_Aug_Vgg,'g',label='Training loss Augmentation VGG16')

plt.plot(epochs,loss_val_Aug_Vgg,'b',label='Validation loss Augmentation VGG16')

plt.title('Training and Validation loss - Augmentation VGG16')



plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(loc='upper right')

plt.show()
# Plotting Val Precision and Recall - Augmentation VGG16



val_precision_Aug_Vgg = data_history_Aug_Vgg.history['val_precision']

val_recall_Aug_Vgg = data_history_Aug_Vgg.history['val_recall']

val_auc_Aug_Vgg = data_history_Aug_Vgg.history['val_auc']

epochs = range(125)



plt.plot(epochs, val_precision_Aug_Vgg, 'c', label='Validation Precision Augmentation VGG16')

plt.plot(epochs, val_recall_Aug_Vgg, 'm', label='Validation Recall Augmentation VGG16')

plt.plot(epochs, val_auc_Aug_Vgg, 'g', label='Validation AUC Augmentation VGG16')

plt.title('Precision, Recall, dan AUC Score - Augmentation VGG16')

plt.xlabel('Epochs')

plt.ylabel('Score')

plt.legend()

plt.show()