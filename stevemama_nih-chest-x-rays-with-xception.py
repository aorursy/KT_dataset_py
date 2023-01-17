import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE

# Create strategy from tpu
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
#tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)
import os
import numpy as np
import pandas as pd
from glob import glob
from itertools import chain
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
!pip install tf-explain
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DATA_DIR = '../input/data/'
image_size = 256
batch_size = 32
df = pd.read_csv(f'{DATA_DIR}Data_Entry_2017.csv')
data_image_paths = {os.path.basename(x): x for x in glob(os.path.join(DATA_DIR, 'images*', '*', '*.png'))}
df['path'] = df['Image Index'].map(data_image_paths.get)
df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if len(x) > 0]
labels
for label in labels:
    if len(label) > 1:
        df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)
df.head()
labels = [label for label in labels if df[label].sum() > 1000]
labels
train_df, valid_df = train_test_split(df, test_size=0.20, random_state=2018, stratify=df['Finding Labels'].map(lambda x: x[:4]))
train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
core_idg = ImageDataGenerator(rescale=1 / 255,
                                  samplewise_center=True,
                                  samplewise_std_normalization=True,
                                  horizontal_flip=True,
                                  vertical_flip=False,
                                  height_shift_range=0.05,
                                  width_shift_range=0.1,
                                  rotation_range=5,
                                  shear_range=0.1,
                                  fill_mode='reflect',
                                  zoom_range=0.15)

train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

valid_gen = core_idg.flow_from_dataframe(dataframe=valid_df,
                                             directory=None,
                                             x_col='path',
                                             y_col='labels',
                                             class_mode='categorical',
                                             batch_size=batch_size,
                                             classes=labels,
                                             target_size=(image_size, image_size))

test_X, test_Y = next(core_idg.flow_from_dataframe(dataframe=valid_df,
                                                       directory=None,
                                                       x_col='path',
                                                       y_col='labels',
                                                       class_mode='categorical',
                                                       batch_size=1024,
                                                       classes=labels,
                                                       target_size=(image_size, image_size)))

'''import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from keras.models import load_model


# from tensorflow.keras.applications import DenseNet121
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import Xception
# import tensorflow.keras.layers as Layers'''
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

with strategy.scope():
    #base_model = Xception(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
    base_model = Xception(include_top = False, weights ='imagenet', input_shape =(256,256,3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation ='relu')(x)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(x)
    output = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
    model = tf.keras.Model(base_model.input, output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('../input/weightsmodel/model.Densenet.h5')

def get_callbacks(model_name):
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    callbacks.append(tensor_board)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'model.{model_name}.h5',
        verbose=1,
        save_best_only=True)
    # erly = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    callbacks.append(checkpoint)
    # callbacks.append(erly)
    return callbacks
'''    base_model = tf.keras.applications.Xception(
    weights='imagenet',
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    pooling='avg')
    next_model = tf.keras.applications.InceptionV3(
    weights ='imagenet',
    input_shape =(299,299,3),
    include_top = False,
    pooling ='avg',
    )
      # model = tf.keras.Sequential([
       # efn.EfficientNetB1(
        #    input_shape=(*IMAGE_SIZE, 3),
         #   weights='imagenet',
          #  include_top=False),
    model = tf.keras.Sequential([
        base_model,
        next_model,
        #L.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'),
        #L.GlobalAveragePooling2D(),
        #L.MaxPooling2D(pool_size =(2,2), strides = None),
        
        L.Dense(512, activation = 'relu'), 
        
        L.Dense(1024, activation = 'relu'),
        #L.Dense(1024, activation = 'relu'),
        L.Dense(len(labels), activation='sigmoid')
    ])
    
model.compile(
    optimizer=tf.keras.optimizers.Adam( learning_rate=1e-4, amsgrad=False), 
    #loss = 'binary_crossentropy',
    loss = get_weighted_loss(pos_weights, neg_weights),
    metrics = ['accuracy']
)
model.summary()
#model.load_weights('../input/weights/efficent_net_b1_trained_weights_1.h5')'''

callbacks = get_callbacks('Densenet')
model.fit(train_gen,
              steps_per_epoch=150,
              validation_data=(test_X, test_Y),
              epochs=50,
              callbacks=callbacks)
#adding test_check number 1
#adding test_check number 2
#adding test_check number 3
#adding test_check number 4
#adding test_check number 5
#adding test_check number 6
#adding test_check number 7
#adding test_check number 8
y_pred = model.predict(test_X)
for label, p_count, t_count in zip(labels,
                                     100 * np.mean(y_pred, 0),
                                     100 * np.mean(test_Y, 0)):
    print('%s: actual: %2.2f%%, predicted: %2.2f%%' % (label, t_count, p_count))
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (idx, c_label) in enumerate(labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), y_pred[:, idx])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('trained_net.png')
print('ROC auc score: {:.3f}'.format(roc_auc_score(test_Y.astype(int), y_pred)))
