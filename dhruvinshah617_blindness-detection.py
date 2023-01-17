import os

import cv2

import random

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, cohen_kappa_score

from keras.models import Model

from keras import optimizers, applications

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input

import tensorflow as tf



from keras.applications import inception_v3

from keras.applications.inception_v3 import InceptionV3

from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor

def seed_everything(seed=42):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    #set_random_seed(0)

seed_everything()



%matplotlib inline

sns.set(style="whitegrid")

warnings.filterwarnings("ignore")
 #This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))



# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import regularizers, optimizers

import os

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
os.chdir('/kaggle/input')

!ls
os.getcwd()
df=pd.read_csv('blindness/extra_info.csv')
df['diagnosis'] = df['diagnosis'].astype('str')

df.describe()
df
df = df.sample(frac=1).reset_index(drop=True)

df.head(10)
x_col="id_code"

y_col="diagnosis"
x_col="id_code"

y_col="diagnosis"



datagen=ImageDataGenerator(rescale=1./255,validation_split=0.15)

train_generator=datagen.flow_from_dataframe(dataframe=df,directory="/kaggle/input/blindness/dataset/",x_col=x_col,y_col=y_col,

                                subset="training",batch_size=16,seed=42,shuffle=True,class_mode="categorical",target_size=(300,300))

valid_generator=datagen.flow_from_dataframe(dataframe=df,directory="/kaggle/input/blindness/dataset/",x_col=x_col,y_col=y_col,

                                subset="validation",batch_size=16,seed=42,shuffle=True,class_mode="categorical",target_size=(300,300))
os.chdir('/kaggle/input/')

testdf=pd.read_csv('../input/aptos2019-blindness-detection/test.csv',dtype=str)

testdf.head()
def append_ext(fn):

    return fn+".png"

testdf["id_code"]=testdf["id_code"].apply(append_ext)

testdf
os.chdir('/kaggle/input/aptos2019-blindness-detection/test_images/')

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(

dataframe=testdf,

directory="/kaggle/input/aptos2019-blindness-detection/test_images/",

x_col="id_code",

y_col=None,

batch_size=32,

seed=42,

shuffle=False,

class_mode=None,

target_size=(300, 300))

import numpy as np

from keras import layers

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

from keras.models import Model, load_model

from keras.preprocessing import image

from keras.utils import layer_utils

from keras.utils.data_utils import get_file

from keras.applications.imagenet_utils import preprocess_input

import pydot

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

import keras

from keras.initializers import glorot_uniform

import scipy.misc

from matplotlib.pyplot import imshow

%matplotlib inline

import tensorflow as tf

import keras.backend as K

def kappa_loss(y_pred, y_true, y_pow=2, eps=1e-10, N=5, bsize=16, name='kappa'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""



 



    with tf.name_scope(name):

        y_true = tf.cast(y_true,dtype=tf.float32)

        repeat_op = tf.cast(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]),dtype=tf.float32)

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.cast((N - 1) ** 2,dtype=tf.float32)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.cast(bsize,dtype=tf.float32))

    

        return nom / (denom + eps)
from keras.models import model_from_json


os.chdir('/kaggle/working')

json_file = open('/kaggle/working/model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
# add new classifier layers

flat1 =  GlobalAveragePooling2D()(loaded_model.output)



class2=Dense(64, activation='relu')(flat1)

output = Dense(5, activation='softmax')(class2)

# define new model

model = Model(inputs=loaded_model.inputs, outputs=output)

# summarize

model.summary()


# instantiating the model in the strategy scope creates the model on the TPU



model.compile(optimizer=keras.optimizers.Adam(),

                  loss=keras.losses.CategoricalCrossentropy(),

                  metrics=[kappa_loss,keras.metrics.CategoricalAccuracy()])

model.fit_generator(train_generator,steps_per_epoch=3000/20  ,epochs=15,

                                  validation_data=valid_generator)

import gc 

gc.collect()

test_generator.reset()

pred=loaded_model.predict_generator(test_generator)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames

results=pd.DataFrame({"id_code":filenames,

                      "diagnosis":predictions})

os.chdir('/kaggle/working')
results.to_csv("submission.csv",index=False)