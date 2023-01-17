# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('../input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow import keras



rsna_df = pd.read_csv("../input/rsna-bone-age/boneage-training-dataset.csv")

base_bone_dir = '../input/rsna-bone-age/'

rsna_df['path'] = rsna_df['id'].map(lambda x: os.path.join(base_bone_dir,

                                                         'boneage-training-dataset', 

                                                         'boneage-training-dataset', 

                                                          '{}.png'))

#rsna_df['imagepath'] = [f'{pid}.png' for pid in rsna_df.id]

rsna_df['imagepath'] = rsna_df['id'].map(lambda x: '{}.png'.format(x))

rsna_df.head()

bone_age_mean = rsna_df['boneage'].mean()

bone_age_dev = 2 * rsna_df['boneage'].std()

bone_age_std = rsna_df['boneage'].std()

rsna_df['bone_age_zscore'] = rsna_df.boneage.map(lambda x: (x - bone_age_mean)/bone_age_dev)

# we take the mean , dev as 0 and 1

#bone_age_mean = 0

#bone_age_dev = 1.0

rsna_df['bone_age_float'] = rsna_df.boneage.map(lambda x: (x - 0.)/1.)

rsna_df.dropna(inplace = True)

rsna_df.head(5)
rsna_df['gender'] = rsna_df['male'].map(lambda x: 'male' if x else 'female')

rsna_df.head()

import seaborn as sns

gender = sns.countplot(rsna_df['gender'])

rsna_df['sex'] = rsna_df['gender'].map(lambda x: 1 if x=='male' else 0)

rsna_df.head()
X = pd.DataFrame(rsna_df[['id','bone_age_float','imagepath','bone_age_zscore']])
Y = pd.DataFrame(X['bone_age_zscore'])
from pathlib import Path

train_img_path = Path('../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/')

#test_img_path = Path('../input/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/')
from sklearn.model_selection import train_test_split

x_train,    x_test,  y_train, y_test = train_test_split(X,Y, 

                                   test_size = 0.2, 

                                   random_state = 2020,

                                   )

print(' x train', x_train.shape[0], 'x validation', x_test.shape[0])

print('y train', y_train.shape[0], 'y validation', y_test.shape[0])
#For training  i have taken only these records

#x_train = x_train.head(9600)

#x_test = x_train.tail(3000)

#y_train = y_train.head(9600)

#y_test = y_train.tail(3000)
import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

import tensorflow as tf
tf.keras.backend.clear_session()
img_rows = 224

img_cols = 224



datagen=ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,

                           width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,

                           horizontal_flip=True, vertical_flip = False, fill_mode="nearest"

                           )



train_gen_mnet=datagen.flow_from_dataframe(dataframe=x_train,

                                            #directory=train_img_path,

                                            directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/", 

                                            x_col='imagepath', 

                                            y_col= 'bone_age_zscore', 

                                            class_mode = 'raw',

                                            color_mode = 'rgb',

                                            target_size = (img_rows, img_cols), 

                                            batch_size=64)

valid_gen_mnet=datagen.flow_from_dataframe(dataframe=x_test,

                                            #directory=train_img_path,

                                            directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/", 

                                            x_col='imagepath', 

                                            y_col= 'bone_age_zscore', 

                                            class_mode = 'raw',

                                            color_mode = 'rgb',

                                            target_size = (img_rows, img_cols), 

                                            batch_size=64)
STEP_SIZE_TRAIN=np.ceil(train_gen_mnet.n//train_gen_mnet.batch_size)

STEP_SIZE_VALID=np.ceil(valid_gen_mnet.n//valid_gen_mnet.batch_size)
print (STEP_SIZE_TRAIN, STEP_SIZE_VALID)
train_img_mnet, train_lbl_mnet = next(train_gen_mnet)

valid_img_mnet, valid_lbl_mnet = next(valid_gen_mnet)
train_img_mnet.shape
train_X, train_Y = next(datagen.flow_from_dataframe(dataframe=x_train, 

                                            directory="../input/rsna-bone-age/boneage-training-dataset/boneage-training-dataset/", 

                                            #directory=train_img_path,

                                            x_col='imagepath', 

                                            y_col='bone_age_zscore', 

                                            class_mode = 'raw',

                                            color_mode = 'rgb',

                                            target_size=(224, 224), 

                                            batch_size=1024))
train_X.shape
base_model=MobileNet(input_shape =  train_img_mnet.shape[1:],weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.



base_mobilenet_model=base_model.output

base_mobilenet_model=GlobalAveragePooling2D()(base_mobilenet_model)

base_mobilenet_model=Dense(1024,activation='relu')(base_mobilenet_model) #we add dense layers so that the model can learn more complex functions and classify for better results.

base_mobilenet_model=Dense(1024,activation='relu')(base_mobilenet_model) #dense layer 2

base_mobilenet_model=Dense(512,activation='relu')(base_mobilenet_model) #dense layer 3

output1=Dense(1,activation='linear')(base_mobilenet_model) #final layer with linear activation
rsna_mobilenet=Model(inputs=base_model.input,outputs=output1)

#specify the inputs

#specify the outputs

#now a model has been created based on our architecture

for i,layer in enumerate(rsna_mobilenet.layers):

  print(i,layer.name)

for layer in rsna_mobilenet.layers:

    layer.trainable=False

# or if we want to set the first 20 layers of the network to be non-trainable

for layer in rsna_mobilenet.layers[:20]:

    layer.trainable=False

for layer in rsna_mobilenet.layers[20:]:

    layer.trainable=True

rsna_mobilenet.compile(optimizer = 'adam', loss = 'mse',

                           metrics = ['mae'])



rsna_mobilenet.summary()
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path="{}_mnet_weights.h5".format('bone_age')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)





reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='auto', min_delta=0.01, cooldown=3, min_lr=0.01)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=3) # probably needs to be more patient

callbacks_list = [checkpoint, early, reduceLROnPlat]
history_mobilenet = rsna_mobilenet.fit_generator( generator=train_gen_mnet,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_gen_mnet,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=10

                    ,callbacks = callbacks_list)
mae = history_mobilenet.history['mae']

val_mae = history_mobilenet.history['val_mae']



loss = history_mobilenet.history['loss']

val_loss = history_mobilenet.history['val_loss']

epochs = 6

epochs_range = range(epochs)



plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, mae, label='Training MAE ')

plt.plot(epochs_range, val_mae, label='Validation MAE ')

plt.legend(loc='lower right')

plt.title('Training and Validation MAE')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label='Training Loss')

plt.plot(epochs_range, val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.title('Training and Validation Loss')

plt.show()
#print ( (rsna_mobilenet.evaluate(valid_gen_mnet, verbose = 0))*100)
from keras.models import Model

import keras.backend as K
rsna_mobilenet.load_weights(weight_path)
val_pred_Y = (bone_age_dev*rsna_mobilenet.predict(train_X, batch_size = 64, verbose = True))+bone_age_mean

val_Y_months = (bone_age_dev*train_Y)+bone_age_mean
rand_idx = np.random.choice(range(train_X.shape[0]), 80)

fig, m_axs = plt.subplots(20, 4, figsize = (14, 30))

for (idx, c_ax) in zip(rand_idx, m_axs.flatten()):

    c_ax.imshow(train_X[idx, :,:,0], cmap = 'bone')

    c_ax.set_title('\n\nActual (Prediction) : %2.1f (%2.1f)' % (val_Y_months[idx], val_pred_Y[idx]))

    c_ax.axis('off')
fig, ax1 = plt.subplots(1,1, figsize = (10,10))

ax1.plot(val_Y_months, val_pred_Y, 'r.', label = 'predictions')

ax1.plot(val_Y_months, val_Y_months, 'b-', label = 'actual')

ax1.legend()

ax1.set_xlabel('Actual Age (Months)')

ax1.set_ylabel('Predicted Age (Months)')
# evaluate the model

_, train_acc = rsna_mobilenet.evaluate(train_gen_mnet)

_, valid_acc = rsna_mobilenet.evaluate(valid_gen_mnet)

print('Train: %.3f, Validation: %.3f' % (train_acc, valid_acc))
test_df = pd.read_csv("../input/rsna-bone-age/boneage-test-dataset.csv")
test_df.head()
test_df.shape
test_img_path = ('../input/rsna-bone-age/boneage-test-dataset/')
test_images = os.listdir(test_img_path)

#print(len(test_images), 'test images found')
pred_datagen = ImageDataGenerator(rescale=1./255)

pred_generator = pred_datagen.flow_from_directory(

        str(test_img_path),#"../input/rsna-bone-age/boneage-test-dataset/boneage-test-dataset/",

        target_size=(224, 224),

        batch_size=10,

        class_mode='sparse',

        color_mode ='rgb',

        shuffle=False)
_, pred_acc = rsna_mobilenet.evaluate(pred_generator)
print('Test: %.3f' % (pred_acc))
img_batch = next(pred_generator)
pred=rsna_mobilenet.predict_generator(pred_generator, steps=len(pred_generator), verbose=1)
# Get classes by np.round

cl = np.round(pred)

# Get filenames (set shuffle=false in generator is important)

filenames=pred_generator.filenames
y_months = (pred[:,0]*41.18 + 127.32).astype(int)
y_months
# Data frame

results=pd.DataFrame({"file":filenames,"prediction":y_months})
results.head(20)
results.to_csv("boneage_testdata_predict.csv")
test_df.insert(2, "Boneage",y_months, True)
test_df.head()