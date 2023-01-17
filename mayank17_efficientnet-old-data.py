import time

t_start = time.time()
import warnings

warnings.filterwarnings("ignore")



import pandas as pd 

import numpy as np

import os, gc, sys

import matplotlib.pyplot as plt

import seaborn as sns



import keras

from keras import backend as k

from keras import layers, models, optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import img_to_array, load_img



from keras.models import Model, load_model

from keras.applications.resnet50 import preprocess_input

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint





sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))

from efficientnet import EfficientNetB5
IMG_WIDTH   = 456

IMG_HEIGHT  = 456

CHANNEL     = 3





# Model parameters

BATCH_SIZE      = 4

EPOCHS_OLD_DATA = 10

WARMUP_EPOCHS   = 3



NUM_CLASSES = 5

SEED        = 2



LEARNING_RATE        = 1e-4

WARMUP_LEARNING_RATE = 1e-3



ES_PATIENCE    = 5

RLROP_PATIENCE = 3

DECAY_DROP     = 0.5
BASE_DIR  = '/kaggle/input/aptos2019-blindness-detection/'

TRAIN_DIR = '/kaggle/input/aptos2019-blindness-detection/train_images'

TEST_DIR  = '/kaggle/input/aptos2019-blindness-detection/test_images'



TRAIN_DIR = '/kaggle/input/diabetic-retinopathy-resized/resized_train/resized_train'
TRAIN_DF = pd.read_csv(BASE_DIR + "train.csv",dtype='object')

TEST_DF = pd.read_csv(BASE_DIR + "test.csv",dtype='object')



# changing columns using .columns() ---> oly required with new data

TRAIN_DF = pd.read_csv("/kaggle/input/diabetic-retinopathy-resized/trainLabels.csv",dtype='object')





X_COL='id_code'

Y_COL='diagnosis'
# changing columns using .columns() ---> oly required with new data

TRAIN_DF.columns = ['id_code', 'diagnosis'] 



# print(TRAIN_DF.head())

# print(TEST_DF.head())
def append_file_ext(file_name):

    return file_name + ".png"





# changing columns using .columns() ---> oly required with new data

def append_file_ext_jpeg(file_name):

    return file_name.replace(".png",".jpeg")
TRAIN_DF[X_COL] = TRAIN_DF[X_COL].apply(append_file_ext)

TEST_DF[X_COL] = TEST_DF[X_COL].apply(append_file_ext)





# changing columns using .columns() ---> oly required with new data

TRAIN_DF[X_COL] = TRAIN_DF[X_COL].apply(append_file_ext_jpeg)
print(TRAIN_DF.head())

print('************************')

print(TEST_DF.head())

print('************************')

print(len(TRAIN_DF))

print('************************')

print(len(TEST_DF))
# TRAIN_DF['diagnosis'].hist(figsize=(10,5), bins=10)

# print(TRAIN_DF['diagnosis'].value_counts())
df0 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '0']

df1 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '1']

df2 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '2']

df3 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '3']

df4 = TRAIN_DF.loc[TRAIN_DF['diagnosis'] == '4']



df0 = df0.head(2000)

df1 = df1.head(2000)

df2 = df2.head(2000)



TRAIN_DF = df0.append([df1, df2, df3, df4],ignore_index = True)

print(TRAIN_DF.head())

print('**********************************')

from sklearn.utils import shuffle

TRAIN_DF = shuffle(TRAIN_DF)



print('**********************************')

print(len(TRAIN_DF))

print(TRAIN_DF.head())
# TRAIN_DF = TRAIN_DF.head(200)
# print(TRAIN_DF.head())

# print('************************')

# print(TEST_DF.head())

# print('************************')

# print(len(TRAIN_DF))

# print('************************')

# print(len(TEST_DF))
def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    

    base_model = EfficientNetB5(weights=None, 

                                       include_top=False,

                                       input_tensor=input_tensor)

    base_model.load_weights('/kaggle/input/efficientnet-keras-weights-b0b5/efficientnet-b5_imagenet_1000_notop.h5')

        



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu')(x)

#     x = Dropout(0.5)(x)

#     x = Dense(1024, activation='relu')(x)

    

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output)

    

    return model
model = create_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNEL), n_out=NUM_CLASSES)



# # Replace all Batch Normalization layers by Group Normalization layers

# for i, layer in enumerate(model.layers):

#     if "batch_normalization" in layer.name:

#         model.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)

        

        

for layer in model.layers:

    layer.trainable = False



for i in range(-7, 0):

    model.layers[i].trainable = True



metric_list = ["accuracy"]

optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)

model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

# model.summary()
def crop_image_from_gray(img, tol=7):

    # If for some reason we only have two channels

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def preprocess_image(path, sigmaX=10):

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (img_width, img_height))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)

    return image
# print(TRAIN_DF.head())

# print(TRAIN_DIR)
datagen         = ImageDataGenerator(

                    rescale=1./255.,

                    validation_split=0.25)



train_generator = datagen.flow_from_dataframe(

                    dataframe=TRAIN_DF,

                    directory=TRAIN_DIR,

                    x_col=X_COL,

                    y_col=Y_COL,

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    class_mode="categorical",

                    preprocessing_function=preprocess_image,

                    target_size=(IMG_WIDTH,IMG_HEIGHT))



valid_generator=datagen.flow_from_dataframe(

                    dataframe=TRAIN_DF,

                    directory=TRAIN_DIR,

                    x_col=X_COL,

                    y_col=Y_COL,

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    zoom_range=0.2,

                    horizontal_flip=True,

                    class_mode="categorical",

                    preprocessing_function=preprocess_image,    

                    target_size=(IMG_WIDTH,IMG_HEIGHT))



test_datagen=ImageDataGenerator(rescale=1./255.)



test_generator=test_datagen.flow_from_dataframe(

                    dataframe=TEST_DF,

                    directory=TEST_DIR,

                    x_col=X_COL,

                    y_col=None,

                    batch_size=BATCH_SIZE,

                    seed=SEED,

                    shuffle=False,

                    class_mode=None,

                    preprocessing_function=preprocess_image,    

                    target_size=(IMG_WIDTH,IMG_HEIGHT))
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size



print(STEP_SIZE_TRAIN)

print(STEP_SIZE_VALID)

print(STEP_SIZE_TEST)
gc.collect()
history_warmup = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=WARMUP_EPOCHS,

                              verbose=1).history
gc.collect()
for layer in model.layers:

    layer.trainable = True



es = EarlyStopping(monitor='val_loss',

                   mode='min', 

                   patience=ES_PATIENCE, 

                   restore_best_weights=True, 

                   verbose=1)



rlrop = ReduceLROnPlateau(monitor='val_loss', 

                          mode='min', 

                          patience=RLROP_PATIENCE, 

                          factor=DECAY_DROP, 

                          min_lr=1e-6, 

                          verbose=1)



#define the model checkpoint callback -> this will keep on saving the model as a physical file

model_checkpoint = ModelCheckpoint('EfficientNetB5_Best_KV.h5',

                                   verbose=1, 

                                   save_best_only=True)



callback_list = [es, rlrop, model_checkpoint]

optimizer = optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=metric_list)

# model.summary()
collected = gc.collect() 

print("Garbage collector: collected","%d objects." % collected) 
history_finetunning = model.fit_generator(generator=train_generator,

                              steps_per_epoch=STEP_SIZE_TRAIN,

                              validation_data=valid_generator,

                              validation_steps=STEP_SIZE_VALID,

                              epochs=EPOCHS_OLD_DATA,

                              callbacks=callback_list,

                              verbose=1).history
gc.collect()
history = {'loss': history_warmup['loss'] + history_finetunning['loss'], 

           'val_loss': history_warmup['val_loss'] + history_finetunning['val_loss'], 

           'acc': history_warmup['acc'] + history_finetunning['acc'], 

           'val_acc': history_warmup['val_acc'] + history_finetunning['val_acc']}



sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))



ax1.plot(history['loss'], label='Train loss')

ax1.plot(history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history['acc'], label='Train Accuracy')

ax2.plot(history['val_acc'], label='Validation accuracy')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()
model = load_model('EfficientNetB5_Best_KV.h5')
if test_generator.n%BATCH_SIZE > 0:

    PREDICTION_STEPS = (test_generator.n//BATCH_SIZE) + 1

else:

    PREDICTION_STEPS = (test_generator.n//BATCH_SIZE)



print(PREDICTION_STEPS)
print(test_generator.n)

print(test_generator.batch_size)

print(STEP_SIZE_TEST)

print(BATCH_SIZE)
test_generator.reset()

preds = model.predict_generator(test_generator,

                                steps=PREDICTION_STEPS, 

                                verbose=1) 

predictions = [np.argmax(pred) for pred in preds]
gc.collect()
filenames = test_generator.filenames

# print(len(filenames))

# print(len(predictions))
results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])

results.astype({'diagnosis': 'int64'})

results.to_csv('submission.csv',index=False)

print(results.head(10))
gc.collect()
# Check kernels run-time. GPU limit for this competition is set to ± 9 hours.

t_finish = time.time()

total_time = round((t_finish-t_start) / 3600, 4)

print('Kernel runtime = {} hours ({} minutes)'.format(total_time, 

                                                      int(total_time*60)))