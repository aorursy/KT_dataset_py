!pip install -q efficientnet
import os



from numpy.random import seed

seed(101)



import numpy as np

import pandas as pd



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras import Sequential

from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import efficientnet.tfkeras as efn 



from sklearn.metrics import confusion_matrix



from matplotlib import pyplot as plt

%matplotlib inline

print("Running TensorflowVersion: " + str(tf.__version__))
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

except ValueError:

    tpu = None

    gpus = tf.config.experimental.list_logical_devices('GPU')



#Select appropriate distribution strategy for hardware



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print("Running on TPU: " + str(tpu.master()))

elif len(gpus) > 0:

    strategy = tf.distribute.MirroredStrategy(gpus)

    print("Running on ",len(gpus)," GPU(s)")

else:

    strategy = tf.distribute.get_strategy()

    print("Running on CPU")
train_path = "/kaggle/input/100-bird-species/train"

val_path = "/kaggle/input/100-bird-species/valid"

test_path = "/kaggle/input/100-bird-species/test"



train_images = [image for dir,_,sublist in os.walk("/kaggle/input/100-bird-species/train") for image in sublist]

val_images = [image for dir,_,sublist in os.walk("/kaggle/input/100-bird-species/valid") for image in sublist]

test_images = [image for dir,_,sublist in os.walk("/kaggle/input/100-bird-species/test") for image in sublist]

num_train_images = len(train_images)

num_val_images = len(val_images)

num_test_images = len(test_images)



IMAGE_SIZE = 224

EPOCHS = 10



#CLASSES = ['Antelope','Bat','Beaver','Bobcat','Buffalo','Chihuahua','Chimpanzee','Collie','Dalmatian','German Shepherd',

#           'Grizzly Bear','Hippopotamus','Horse','Killer Whale','Mole','Moose','Mouse','Otter','Ox','Persian Cat','Raccoon',

#           'Rat','Rhinoceros','Seal','Siamese Cat','Spider Monkey','Squirrel','Walrus','Weasel','Wolf']



#Learning rate scheduling variables

num_units = strategy.num_replicas_in_sync

if num_units == 8:

    BATCH_SIZE = 16 * num_units

    VALIDATION_BATCH_SIZE = 16 * num_units

    start_lr = 0.00001

    min_lr = 0.00001

    max_lr = 0.00005 * num_units

    rampup_epochs = 8

    sustain_epochs = 0

    exp_decay = 0.8

elif num_units == 1:

    BATCH_SIZE = 16

    VALIDATION_BATCH_SIZE = 16

    start_lr = 0.00001

    min_lr = 0.00001

    max_lr = 0.0002

    rampup_epochs = 8

    sustain_epochs = 0

    exp_decay = 0.8

else:

    BATCH_SIZE = 8 * num_units

    VALIDATION_BATCH_SIZE = 8 * num_units

    start_lr = 0.00001

    min_lr = 0.00001

    max_lr = 0.00002 * num_units

    rampup_epochs = 11

    sustain_epochs = 0

    exp_decay = 0.8

    

train_steps = int(np.ceil(num_train_images/BATCH_SIZE))

val_steps = int(np.ceil(num_val_images/VALIDATION_BATCH_SIZE))



print("Total Training Images: " + str(num_train_images))

print("Total Validation Images: " + str(num_val_images))

print("Total Test Images: " + str(num_test_images))



print("Train Steps: " + str(train_steps))

print("Val steps: " + str(val_steps))
def display_training_curves(training,validation,title,subplot):

    if subplot%10 == 1:

        plt.subplots(figsize = (10,10),facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])

    

def learningrate_function(epoch):

    if epoch < rampup_epochs:

        lr = (max_lr - start_lr)/rampup_epochs * epoch + start_lr

    elif epoch < rampup_epochs + sustain_epochs:

        lr = max_lr

    else:

        lr = (max_lr - min_lr) * exp_decay**(epoch - rampup_epochs - sustain_epochs) + min_lr

    return lr



def learning_rate_callback():

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch : learningrate_function(epoch),verbose = True)

    rng = [i for i in range(EPOCHS)]

    y = [learningrate_function(x) for x in range(EPOCHS)]

    plt.plot(rng,y)

    return lr_callback
datagen = ImageDataGenerator(rescale = 1.0/255)



train_datagen = datagen.flow_from_directory(train_path, 

                                            target_size = (IMAGE_SIZE,IMAGE_SIZE),

                                            batch_size = BATCH_SIZE, 

                                            class_mode = 'categorical')



val_datagen = datagen.flow_from_directory(val_path,

                                          target_size = (IMAGE_SIZE,IMAGE_SIZE),

                                          batch_size = VALIDATION_BATCH_SIZE,

                                          class_mode = 'categorical')



test_datagen = datagen.flow_from_directory(val_path,

                                            target_size = (IMAGE_SIZE,IMAGE_SIZE),

                                            batch_size = 1, 

                                            class_mode = 'categorical',

                                            shuffle = False)
CLASSES = list(train_datagen.class_indices.keys())
len(CLASSES)
with strategy.scope():

    enet = efn.EfficientNetB0(

    input_shape = (IMAGE_SIZE,IMAGE_SIZE,3),

    weights = 'imagenet',

    include_top = False)

    enet.trainable = True

    

    model = Sequential([

        enet,

        GlobalAveragePooling2D(),

        Dense(200, activation = 'softmax', dtype= tf.float32)

    ])

    

    model.compile(

    optimizer = 'adam',

    loss = 'categorical_crossentropy',

    metrics = ['accuracy']

    )

    

model.summary()
lr_callback = learning_rate_callback()
filepath = "my_model_bird.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')



callbacks_list = [checkpoint,lr_callback]



hist = model.fit_generator(train_datagen,

                             steps_per_epoch = train_steps,

                             validation_data = val_datagen,

                             validation_steps = val_steps,

                             epochs = EPOCHS,

                             verbose = 1,

                             callbacks = callbacks_list)
model.save("my_model_bird_manual.h5")
model = load_model("my_model_bird.hdf5")

model.summary()
val_loss, val_acc = model.evaluate_generator(val_datagen,steps = num_val_images)



print('val_loss:', val_loss)

print('val_acc:', val_acc)
display_training_curves(hist.history['loss'], hist.history['val_loss'], 'loss', 211) 

display_training_curves(hist.history['accuracy'], hist.history['val_accuracy'], 'accuracy', 212)
print(val_datagen.classes)
predictions = model.predict_generator(test_datagen, steps= num_val_images, verbose=1)
predictions.shape
predictions.argmax(axis=1)
df_preds = pd.DataFrame(predictions,columns=CLASSES)
df_preds.head()
y_true = val_datagen.classes

y_pred = df_preds['ALBATROSS']

print(y_true)

print(y_pred)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
val_labels = val_datagen.classes



cm = confusion_matrix(val_labels, predictions.argmax(axis=1))



cm_plot_labels = CLASSES



plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
from sklearn.metrics import classification_report



# Generate a classification report



# For this to work we need y_pred as binary labels not as probabilities

y_pred_binary = predictions.argmax(axis=1)



report = classification_report(y_true, y_pred_binary, target_names=cm_plot_labels)



print(report)