# General Libs
from tensorflow import keras
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import *
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

im_shape = (150,150)
TRAINING_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'
TEST_DIR = '/kaggle/input/chest-xray-pneumonia/chest_xray/test'

seed = random.randint(1, 1000)

BATCH_SIZE = 32
num_classes = 2

learning_rate = 0.0001
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.33) #rescale=1./255,

train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(im_shape[0],im_shape[1]), shuffle=True, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")#, color_mode='grayscale')

validation_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(im_shape[0],im_shape[1]), shuffle=False, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")#, color_mode='grayscale')
nb_train_samples = train_generator.samples
nb_validation_samples = validation_generator.samples
from keras.preprocessing.image import ImageDataGenerator

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)#rescale=1./255

test_generator = test_generator.flow_from_directory(TEST_DIR, target_size=(im_shape[0],im_shape[1]), shuffle=False, seed=seed,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")#, color_mode='grayscale')
nb_test_samples = test_generator.samples

# Looking for some examples

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
img = train_generator.filepaths[np.random.randint(low=0, high=train_generator.samples)]
print(img)
img = mpimg.imread(img)
plt.imshow(img);

plt.subplot(1, 2, 2)
img = test_generator.filepaths[np.random.randint(low=0, high=test_generator.samples)]
print(img)
img = mpimg.imread(img)
plt.imshow(img);
# Loading MobileNet without top layer
pre_trained_weights = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = InceptionV3(weights=pre_trained_weights, include_top=False, input_shape=(im_shape[0], im_shape[1], 3))

x = base_model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freezing pretrained layers
for layer in base_model.layers:
    layer.trainable=False
    
#optimizer = Adam(lr=learning_rate)
optimizer = keras.optimizers.RMSprop(lr=0.0001)
model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes), 
            train_generator.classes)
class_weights
epochs = 10

#Salvar o melhor modelo
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='model.h5',
        monitor='val_loss', save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=50,verbose=1)
]

history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // BATCH_SIZE,
        epochs=epochs,
        callbacks = callbacks_list,
        validation_data=validation_generator,
        verbose = 1,
        validation_steps=nb_validation_samples // BATCH_SIZE,
        class_weight = class_weights)


#Vamos ver como foi o treino?
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs_x = range(1, len(loss_values) + 1)
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(epochs_x, loss_values, 'bo', label='Training loss')
plt.plot(epochs_x, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
#plt.legend()
plt.subplot(2,1,2)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs_x, acc_values, 'bo', label='Training acc')
plt.plot(epochs_x, val_acc_values, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
from tensorflow.keras.models import load_model
# Load the best saved model
model = load_model('model.h5')
# Vendo alguns reports
# Usando sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(test_generator, nb_test_samples // BATCH_SIZE+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['Normal', 'Pneumo']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

