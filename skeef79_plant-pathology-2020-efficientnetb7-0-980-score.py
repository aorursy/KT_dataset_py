import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import tensorflow as tf, tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers

from kaggle_datasets import KaggleDatasets
AUTO = tf.data.experimental.AUTOTUNE

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



GCS_DS_PATH = KaggleDatasets().get_gcs_path()
IMG_SIZE = 784

BATCH_SIZE = 8*strategy.num_replicas_in_sync

nb_classes = 4
path='../input/plant-pathology-2020-fgvc7/'



train = pd.read_csv(path+'train.csv')

train_id = train['image_id']

train.pop('image_id')



y_train = train.to_numpy().astype('float32')

category_names = ['healthy','multiple_diseases','rust','scab']



root = 'images'

images_paths = [(os.path.join(GCS_DS_PATH,root,idee+'.jpg')) for idee in train_id]
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(images_paths,y_train,test_size=0.2,shuffle=True)


from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',np.unique(y_train.argmax(axis=1)),y_train.argmax(axis=1))

print('class weights: ',class_weights)



plt.bar(range(4),1/class_weights,color=['springgreen', 'lightcoral', 'mediumpurple', 'gold'],width=0.9)

plt.xticks(range(4), category_names) 



plt.title("Categories distribution");

plt.ylabel('Probability')

plt.xlabel('Data')

plt.show()



#class weights to dict

c_w = dict(zip(range(4),class_weights))
def decode_image(filename, label=None, image_size=(IMG_SIZE, IMG_SIZE)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    #convert to numpy and do some cv2 staff mb?

    

    if label is None:

        return image

    else:

        return image, label



def data_augment(image, label=None, seed=5050):

    image = tf.image.random_flip_left_right(image, seed=seed)

    image = tf.image.random_flip_up_down(image, seed=seed)

           

    if label is None:

        return image

    else:

        return image, label
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_augment, num_parallel_calls=AUTO)

    .repeat()

    .shuffle(512)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

    )
val_dataset = (tf.data.Dataset

               .from_tensor_slices((x_val,y_val))

               .map(decode_image,num_parallel_calls=AUTO)

               .batch(BATCH_SIZE)

               .cache()

               .prefetch(AUTO)

              )
!pip install efficientnet

import efficientnet.tfkeras as efn

import tensorflow as tf, tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers
def get_model():

    base_model = efn.EfficientNetB7(weights='imagenet',

                          include_top=False,

                          input_shape=(IMG_SIZE,IMG_SIZE, 3),

                          pooling='avg')

    x = base_model.output

    predictions = Dense(nb_classes, activation="softmax")(x)

    return Model(inputs=base_model.input, outputs=predictions)
from tensorflow.keras.optimizers import Adam



with strategy.scope():

    model = get_model()



opt = Adam(lr=0.001)

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.callbacks import ReduceLROnPlateau



model_name = 'effNetPlants.h5'



#good callbacks

best_model = ModelCheckpoint(model_name, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True,mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5,verbose=1,min_lr=0.000001,patience=6)
history = model.fit(train_dataset,

                    steps_per_epoch=y_train.shape[0]//BATCH_SIZE,

                    epochs=5,

                    verbose=1,

                    validation_data=val_dataset,

                    callbacks=[reduce_lr,best_model]

                    )
plt.title('model accuracy')

plt.plot(history.history['val_accuracy'])

plt.plot(history.history['accuracy'])

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
path='../input/plant-pathology-2020-fgvc7/'



test = pd.read_csv(path+'test.csv')

test_id = test['image_id']



root = 'images'

x_test = [(os.path.join(GCS_DS_PATH,root,idee+'.jpg')) for idee in test_id]
model.load_weights(model_name)
test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .map(decode_image, num_parallel_calls=AUTO)

    .batch(BATCH_SIZE)

)
y_pred = model.predict(test_dataset,verbose=1)
def save_results(y_pred):

    

    path='../input/plant-pathology-2020-fgvc7/'

    test = pd.read_csv(path + 'test.csv')

    test_id = test['image_id']



    res = pd.read_csv(path+'train.csv')

    res['image_id'] = test_id

  

    labels = res.keys()



    for i in range(1,5):

        res[labels[i]] = y_pred[:,i-1]



    res.to_csv('submission.csv',index=False)

  

    print(res.head)
save_results(y_pred)