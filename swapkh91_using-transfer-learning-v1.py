!pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git

from learntools.core import binder

binder.bind(globals())

from learntools.deep_learning.ex_tpu import *

step_1.check()
from petal_helper import *

from tensorflow.keras import layers

import seaborn as sns

from matplotlib import pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
# Detect TPU, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
ds_train = get_training_dataset()

ds_valid = get_validation_dataset()

ds_test = get_test_dataset()



print("Training:", ds_train)

print ("Validation:", ds_valid)

print("Test:", ds_test)
# check if the pixel values are in `[0,1]`.

image_batch, labels_batch = next(iter(ds_train))

first_image = image_batch[0]



print(np.min(first_image), np.max(first_image)) 
print("Number of classes: {}".format(len(CLASSES)))



print("First five classes, sorted alphabetically:")

for name in sorted(CLASSES)[:5]:

    print(name)



print ("Number of training images: {}".format(NUM_TRAINING_IMAGES))
print("Training data shapes:")

for image, label in ds_train.take(3):

    print(image.numpy().shape, label.numpy().shape)

print("Training data label examples:", label.numpy())
print("Test data shapes:")

for image, idnum in ds_test.take(3):

    print(image.numpy().shape, idnum.numpy().shape)

print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
one_batch = next(iter(ds_train.unbatch().batch(20)))

display_batch_of_images(one_batch)
# BATCH_SIZE = 128 * strategy.num_replicas_in_sync

# WARMUP_EPOCHS = 3

# WARMUP_LEARNING_RATE = 1e-4 * strategy.num_replicas_in_sync

# EPOCHS = 30

# LEARNING_RATE = 3e-5 * strategy.num_replicas_in_sync

# HEIGHT = 512

# WIDTH = 512

# CHANNELS = 3

# N_CLASSES = 104

# ES_PATIENCE = 6

# RLROP_PATIENCE = 3

# DECAY_DROP = 0.3



# model_path = 'DenseNet201_%sx%s.h5' % (HEIGHT, WIDTH)
# def create_model(input_shape, N_CLASSES):

#     base_model = tf.keras.applications.DenseNet201(weights='imagenet', 

#                                           include_top=False,

#                                           input_shape=input_shape)



#     base_model.trainable = False # Freeze layers

#     model = tf.keras.Sequential([

#         base_model,

#         layers.GlobalAveragePooling2D(),

#         layers.Dense(N_CLASSES, activation='softmax')

#     ])

    

#     return model
# with strategy.scope():

#     model = create_model((512, 512, 3), N_CLASSES)

    

# metric_list = ['sparse_categorical_accuracy']



# optimizer = tf.keras.optimizers.Adam(lr=WARMUP_LEARNING_RATE)

# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric_list)

# model.summary()
# STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

# warmup_history = model.fit(x=ds_train, 

#                            steps_per_epoch=STEPS_PER_EPOCH, 

#                            validation_data=ds_valid,

#                            epochs=WARMUP_EPOCHS, 

#                            verbose=2).history
# LR_START = 0.00000001

# LR_MIN = 0.000001

# LR_MAX = LEARNING_RATE

# LR_RAMPUP_EPOCHS = 3

# LR_SUSTAIN_EPOCHS = 0

# LR_EXP_DECAY = .8



# def lrfn(epoch):

#     if epoch < LR_RAMPUP_EPOCHS:

#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

#     elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

#         lr = LR_MAX

#     else:

#         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

#     return lr

    

# rng = [i for i in range(EPOCHS)]

# y = [lrfn(x) for x in rng]



# sns.set(style="whitegrid")

# fig, ax = plt.subplots(figsize=(20, 6))

# plt.plot(rng, y)

# print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))

# for layer in model.layers:

#     layer.trainable = True # Unfreeze layers



# checkpoint = ModelCheckpoint(model_path, monitor='val_loss', mode='min', save_best_only=True)

# es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, 

#                    restore_best_weights=True, verbose=1)

# lr_callback = LearningRateScheduler(lrfn, verbose=1)



# callback_list = [checkpoint, es, lr_callback]



# optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE)

# model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric_list)

# model.summary()
# history = model.fit(x=ds_train, 

#                     steps_per_epoch=STEPS_PER_EPOCH, 

#                     validation_data=ds_valid,

#                     callbacks=callback_list,

#                     epochs=EPOCHS, 

#                     verbose=1)
with strategy.scope():

    pretrained_model = tf.keras.applications.DenseNet201(

        weights='imagenet',

        include_top=False ,

        input_shape=[*IMAGE_SIZE, 3]

    )

    for layer in pretrained_model.layers[:703]:

        layer.trainable = False

    

    model = tf.keras.Sequential([

        # To a base pretrained on ImageNet to extract features from images...

        pretrained_model,

        # ... attach a new head to act as a classifier.

        tf.keras.layers.GlobalAveragePooling2D(),

#         tf.keras.layers.MaxPool2D((2,2) , strides = 2),

#         tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    model.compile(

        optimizer='adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy'],

    )



model.summary()
from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.6, min_lr=0.000001)
# Define the batch size. This will be 16 with TPU off and 128 with TPU on

BATCH_SIZE = 128 * strategy.num_replicas_in_sync



# Define training epochs for committing/submitting. (TPU on)

EPOCHS = 50

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



history = model.fit(

    ds_train,

    validation_data=ds_valid,

    epochs=EPOCHS,

    steps_per_epoch=STEPS_PER_EPOCH,

    callbacks = [learning_rate_reduction]

)
display_training_curves(

    history.history['loss'],

    history.history['val_loss'],

    'loss',

    211,

)

display_training_curves(

    history.history['sparse_categorical_accuracy'],

    history.history['val_sparse_categorical_accuracy'],

    'accuracy',

    212,

)
cmdataset = get_validation_dataset(ordered=True)

images_ds = cmdataset.map(lambda image, label: image)

labels_ds = cmdataset.map(lambda image, label: label).unbatch()



cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()

cm_probabilities = model.predict(images_ds)

cm_predictions = np.argmax(cm_probabilities, axis=-1)



labels = range(len(CLASSES))

cmat = confusion_matrix(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

)

cmat = (cmat.T / cmat.sum(axis=1)).T # normalize
score = f1_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

precision = precision_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

recall = recall_score(

    cm_correct_labels,

    cm_predictions,

    labels=labels,

    average='macro',

)

display_confusion_matrix(cmat, score, precision, recall)
dataset = get_validation_dataset()

dataset = dataset.unbatch().batch(20)

batch = iter(dataset)
images, labels = next(batch)

probabilities = model.predict(images)

predictions = np.argmax(probabilities, axis=-1)

display_batch_of_images((images, labels), predictions)
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)
print('Generating submission.csv file...')



# Get image ids from test set and convert to integers

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')



# Write the submission file

np.savetxt(

    'submission.csv',

    np.rec.fromarrays([test_ids, predictions]),

    fmt=['%s', '%d'],

    delimiter=',',

    header='id,label',

    comments='',

)



# Look at the first few predictions

!head submission.csv