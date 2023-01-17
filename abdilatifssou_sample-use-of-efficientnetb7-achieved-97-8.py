import seaborn as sns
import numpy as np
import pandas as pd
import random
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt 
import glob as gb
from kaggle_datasets import KaggleDatasets
!pip install -q efficientnet
import efficientnet.tfkeras as efn
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
train_df = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"
size = []
files = gb.glob(pathname= str("../input/plant-pathology-2020-fgvc7/images/*.jpg"))
for file in files: 
    image = plt.imread(file)
    size.append(image.shape)
pd.Series(size).value_counts()
# Because it take looong time the output is below
# (1365, 2048, 3)    3620
# (2048, 1365, 3)      22
train_df.head(10)
# train_df.shape
fig,ax=plt.subplots(2,2,figsize=(14,14))
sns.barplot(x=train_df.healthy.value_counts().index,y=train_df.healthy.value_counts(),ax=ax[0,0])
ax[0,0].set_xlabel('Healthy',size=9)
ax[0,0].set_ylabel('Count',size=9)

sns.barplot(x=train_df.multiple_diseases.value_counts().index,y=train_df.multiple_diseases.value_counts(),ax=ax[0,1])
ax[0,1].set_xlabel('Multiple Diseases',size=9)
ax[0,1].set_ylabel('Count',size=9)

sns.barplot(x=train_df.rust.value_counts().index,y=train_df.rust.value_counts(),ax=ax[1,0])
ax[1,0].set_xlabel('Rust',size=9)
ax[1,0].set_ylabel('Count',size=9)

sns.barplot(x=train_df.scab.value_counts().index,y=train_df.scab.value_counts(),ax=ax[1,1])
ax[1,1].set_xlabel('Scab',size=9)
ax[1,1].set_ylabel('Count',size=9)
healthy = list(train_df[train_df["healthy"]==1].image_id)
multiple_diseases = list(train_df[train_df["multiple_diseases"]==1].image_id)
rust = list(train_df[train_df["rust"]==1].image_id)
scab = list(train_df[train_df["scab"]==1].image_id)
def load_image(filenames):
    sample = random.choice(filenames)
    image = load_img("../input/plant-pathology-2020-fgvc7/images/"+sample+".jpg")
    plt.imshow(image) 
load_image(healthy)
load_image(multiple_diseases)
load_image(rust)
load_image(scab)
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
#to verify your dir
!gsutil ls $GCS_DS_PATH
def format_path_gcs(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'
#===============================================================
#===============================================================
X = train_df.image_id.apply(format_path_gcs).values
y = np.float32(train_df.loc[:, 'healthy':'scab'].values)

X_train, X_val, y_train, y_val =train_test_split(X, y, test_size=0.1, random_state=43)
print('done!')
print('Shape of X_train : ',X_train.shape)
print('Shape of y_train : ',y_train.shape)
print('===============================================')
print('Shape of X_val : ',X_val.shape)
print('Shape of y_val : ',y_val.shape)
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
BATCH_SIZE = 4 * strategy.num_replicas_in_sync
STEPS_PER_EPOCH = y_train.shape[0] // BATCH_SIZE
def decode_image(filename, label=None, image_size=(1024,1024)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    
    
    if label is None:
        return image
    else:
        return image, label
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(decode_image,num_parallel_calls=AUTO)
    .map(data_augment,num_parallel_calls=AUTO)
    .repeat()
    .shuffle(256)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(decode_image,num_parallel_calls=AUTO)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
def build_lrfn(lr_start=0.00001, lr_max=0.00005,lr_min=0.00001, lr_rampup_epochs=5,lr_sustain_epochs=0, lr_exp_decay=.8):
    
    lr_max = lr_max * strategy.num_replicas_in_sync
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *\
                 lr_exp_decay**(epoch - lr_rampup_epochs- lr_sustain_epochs) + lr_min
        return lr
    return lrfn
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
EarlyStopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,verbose=True, mode="min")
# reduce_lr =  tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10,
#   verbose = 0, mode = "auto", epsilon = 1e-04, cooldown = 0,
#   min_lr = 1e-5)
# es = tf.keras.callbacks.EarlyStopping(monitor = "val_loss" , verbose = 1 , mode = 'min' , patience = 10 )
#weights='noisy-student',
def Eff_B7_NS():
    model_EfficientNetB7_NS = Sequential([efn.EfficientNetB7(input_shape=(1024,1024,3),weights='noisy-student',include_top=False),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(128,activation='relu'),
                                 tf.keras.layers.Dense(64,activation='relu'),
                                 tf.keras.layers.Dense(4,activation='softmax')])               
    model_EfficientNetB7_NS.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics=['categorical_accuracy'])
    
    
    return model_EfficientNetB7_NS
with strategy.scope():
    model_Eff_B7_NS=Eff_B7_NS()
    
model_Eff_B7_NS.summary()
#del model_Eff_B7_NS
EfficientNetB7_NS = model_Eff_B7_NS.fit(train_dataset,
                    epochs=50,
                    callbacks=[lr_schedule,EarlyStopping],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)
plt.figure()
fig,(ax1, ax2)=plt.subplots(1,2,figsize=(19,7))
ax1.plot(EfficientNetB7_NS.history['loss'])
ax1.plot(EfficientNetB7_NS.history['val_loss'])
ax1.legend(['training','validation'])
ax1.set_title('loss')
ax1.set_xlabel('epoch')

ax2.plot(EfficientNetB7_NS.history['categorical_accuracy'])
ax2.plot(EfficientNetB7_NS.history['val_categorical_accuracy'])
ax2.legend(['training','validation'])
ax2.set_title('Acurracy')
ax2.set_xlabel('epoch')
