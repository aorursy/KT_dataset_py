%matplotlib inline



import datetime as dt



import sys, importlib



from functions_py import * # MODIFIED for kaggle (replaced by functions_py instead of functions)

importlib.reload(sys.modules['functions_py']) # MODIFIED for kaggle



#from display_factorial import *

#importlib.reload(sys.modules['display_factorial'])



import datetime as dt



import sys, importlib



from functions_py import * # MODIFIED for kaggle (replaced by functions_py instead of functions)

importlib.reload(sys.modules['functions_py']) # MODIFIED for kaggle



#from display_factorial import *

#importlib.reload(sys.modules['display_factorial'])



import pandas as pd



pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)



import datetime as dt



import os

import zipfile

import urllib



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np    



import ipywidgets as widgets



import qgrid



import glob



from pandas.plotting import scatter_matrix



from sklearn.model_selection import StratifiedShuffleSplit





from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.metrics import pairwise_distances



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



from sklearn.metrics import classification_report



from sklearn.metrics import confusion_matrix



#from yellowbrick.classifier import ROCAUC

from sklearn.metrics import roc_auc_score



import codecs





DATA_PATH = os.path.join("../input", "stanford-dogs-dataset", "images") # Modified for kaggle

DATA_PATH = os.path.join(DATA_PATH, "Images")



MODEL_OUTPUT_PATH = "/kaggle/working/model.h5"

# You need to manually download file from output of previous session (in MODEL_OUPUT_PATH) and manually upload it in MODEL_INPUT_PATH

# Select "Add data", add dataset, and use "model output from previsous session" as your dataset title

MODEL_INPUT_PATH = '/kaggle/input/model-output-from-previous-session/model.h5'



MODEL_INPUT_PATH_NEWFORMAT = '/kaggle/input/pj7-model/model_endsave' # To change: remove dataset "PJ7 model" and replace by new one

MODEL_OUTPUT_PATH_NEWFORMAT_CHECKPOINTSAVE = "/kaggle/working/model_checkpointsave"

MODEL_OUTPUT_PATH_NEWFORMAT_ENDSAVE = "/kaggle/working/model_endsave"





LOAD_MODEL_FROM_PREVIOUS_NOTEBOOK_RUN = True # If True: then set LOAD_MODE_FROM_VGG16 to False, and set TRAIN_MODEL to either True or False

LOAD_MODEL_FROM_VGG16 = False  # If True:  then set TRAIN_MODEL to True (you want to train at least once, at your first notebook run)

TRAIN_MODEL= False





DATA_PATH_FILE = os.path.join(DATA_PATH, "*.csv")

ALL_FILES_LIST = glob.glob(DATA_PATH_FILE)



ALL_FEATURES = []



plt.rcParams["figure.figsize"] = [16,9] # Taille par défaut des figures de matplotlib



import seaborn as sns

from seaborn import boxplot

sns.set()



#import common_functions



####### Paramètres pour sauver et restaurer les modèles :

import pickle

####### Paramètres à changer par l'utilisateur selon son besoin :





from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error





### For progress bar :

#from tqdm import tqdm_notebook as tqdm  #Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`

from tqdm.notebook import tqdm



# Statsmodel : 

import statsmodels.formula.api as smf



import statsmodels.api as sm

from scipy import stats



from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer
!ls -l {MODEL_OUTPUT_PATH_NEWFORMAT}
MODEL_INPUT_PATH_NEWFORMAT
!ls {MODEL_INPUT_PATH_NEWFORMAT}
from PIL import Image

from io import BytesIO
from keras.applications.vgg16 import VGG16

from keras.layers import Dense





from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Flatten



from keras.preprocessing.image import load_img, img_to_array

from keras.applications.vgg16 import preprocess_input



from keras.applications.vgg16 import decode_predictions



import keras
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = ""

#os.environ["AUTOGRAPH_VERBOSITY"] = "10"

#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"



from platform import python_version

import warnings

import time

import datetime as dt

from sklearn.metrics import classification_report, confusion_matrix

import multiprocessing as mp

import shutil



import matplotlib.pyplot as plt

import matplotlib.image as mpimg



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions

from tensorflow.keras.models import *

from tensorflow.keras.layers import *

from tensorflow.keras.optimizers import *

from tensorflow.keras.utils import *

from tensorflow.keras.callbacks import *

from tensorflow.keras.initializers import *



import pandas as pd

import numpy as np

import seaborn as sn



from PIL import Image

import xml.etree.ElementTree as ET

import psutil

import random



warnings.filterwarnings("ignore")

%matplotlib inline



print("py", python_version())

print("tf", tf.__version__)

print("keras", tf.keras.__version__)

mem = psutil.virtual_memory()

print("mem", mem.total/1024/1024)

cpu = mp.cpu_count()

print("cpu", cpu)



#%system nvidia-smi

#%system rocm-smi
epochs = 50

batch_size = 100

testsplit = .2

targetx = 224

targety = 224

learning_rate = 0.0001

classes = 120

seed = random.randint(1, 1000)



#data_dir = "/kaggle/input/stanford-dogs-dataset/images/Images/"

#annotations_dir = "/kaggle/input/stanford-dogs-dataset/annotations/Annotation/"

cropped_dir = "/kaggle/input/openclassrooms-pj7-modelisation-part-4/cropped"



NB_CLASSES = 120
tf.__version__
datagen = ImageDataGenerator(

        shear_range=0.1,

        zoom_range=0.1,

        brightness_range=[0.9,1.1],

        horizontal_flip=True,

        validation_split=testsplit,

        preprocessing_function=preprocess_input

)



train_generator = datagen.flow_from_directory(

        cropped_dir,

        target_size=(targetx, targety),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=True,

        seed=seed,

        subset="training"

)



test_generator = datagen.flow_from_directory(

        cropped_dir,

        target_size=(targetx, targety),

        batch_size=batch_size,

        class_mode='categorical',

        shuffle=False,

        seed=seed,

        subset="validation"

)
len(test_generator.filepaths)
breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))
breed_list
test_generator.reset() 
if (LOAD_MODEL_FROM_VGG16 == True):

    # Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected

    model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))



    # Récupérer la sortie de ce réseau

    x = Flatten()(model.output)

    #x = model.output # Code from OC training (there was no flatten, didn't work : ValueError: Shapes (None, 20) and (None, 7, 7, 20) are incompatible)



    '''

    x = model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(1024,activation='relu')(x)

    x = Dense(1024,activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(512,activation='relu')(x)

    '''



    # Ajouter la nouvelle couche fully-connected pour la classification à NB_CLASSES classes

    predictions = Dense(NB_CLASSES, activation='softmax')(x)



    # Définir le nouveau modèle

    new_model = keras.Model(inputs=model.input, outputs=predictions)

    

    # Ne pas entraîner les 5 premières couches (les plus basses) 

    #for layer in new_model.layers[:5]:

    #   layer.trainable = False



    for layer in new_model.layers[:-5]:

        layer.trainable=False

    for layer in new_model.layers[-5:]:

        layer.trainable=True

        

    # Compiler le modèle 

    new_model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    

    new_model.summary()

    

    # Same code as the one in LOAD_MODEL_FROM_PREVIOUS_NOTEBOOK_RUN block, except we don't load model from file

    

    #checkpoint = ModelCheckpoint(MODEL_OUTPUT_PATH_NEWFORMAT_CHECKPOINTSAVE, monitor='loss', verbose=1, save_best_only=True, mode='min')

    #callbacks_list = [checkpoint]
#model_info = new_model.fit(images_train, images_train_labels, validation_split=0.20, epochs=50, batch_size=64, verbose=2)
if (LOAD_MODEL_FROM_PREVIOUS_NOTEBOOK_RUN == True):

    !ls -l {MODEL_INPUT_PATH_NEWFORMAT}

    

    #checkpoint = ModelCheckpoint(MODEL_OUTPUT_PATH_NEWFORMAT_CHECKPOINTSAVE, monitor='loss', verbose=1, save_best_only=True, mode='min')

    #callbacks_list = [checkpoint]

    

    new_model = load_model(MODEL_INPUT_PATH_NEWFORMAT)
new_model.get_config()
%%time

#epochs = 1

#TRAIN_MODEL = True



if (TRAIN_MODEL == True):

    params = new_model.fit_generator(generator=train_generator, 

                                    steps_per_epoch=len(train_generator), 

                                    validation_data=test_generator, 

                                    validation_steps=len(test_generator),

                                    epochs=epochs,)

                                    #callbacks=callbacks_list)

    

    plt.subplot(1, 2, 1)

    plt.title('Training and test accuracy')

    plt.plot(params.epoch, params.history['accuracy'], label='Training accuracy')

    plt.plot(params.epoch, params.history['val_accuracy'], label='Test accuracy')

    plt.legend()



    plt.subplot(1, 2, 2)

    plt.title('Training and test loss')

    plt.plot(params.epoch, params.history['loss'], label='Training loss')

    plt.plot(params.epoch, params.history['val_loss'], label='Test loss')

    plt.legend()



    plt.show()

    

    # The history of training curves it not saved by keras

    # We could save it by implementing this: https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object

if (TRAIN_MODEL == True):

    new_model.save(MODEL_OUTPUT_PATH_NEWFORMAT_ENDSAVE)
!date

!ls {MODEL_OUTPUT_PATH_NEWFORMAT_ENDSAVE}
#!ls -l {MODEL_INPUT_PATH}

#!ls -l {MODEL_OUTPUT_PATH}
#if (TRAIN_MODEL == True):

#    !cp {MODEL_INPUT_PATH} {MODEL_OUTPUT_PATH}
#!ls -l {MODEL_OUTPUT_PATH}
# Randomly test an image from the test set



# model.load_weights('dog_breed_classifier.h5')





imageno=np.random.random_integers(low=0, high=test_generator.samples)



name = test_generator.filepaths[imageno]

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(test_generator.filepaths[imageno]).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])
test_generator.reset() 
%time

predictions = new_model.predict_generator(test_generator, steps=len(test_generator)) 
%time

y = np.argmax(predictions, axis=1)
%time

print('Classification Report')

cr = classification_report(y_true=test_generator.classes, y_pred=y, target_names=test_generator.class_indices)

print(cr)
test_generator.classes
len(test_generator.classes)
test_generator.class_indices
classnum_toclasslabel = dict(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))
test_labels = [classnum_toclasslabel[k] for k in test_generator.classes]
predicted_test_labels = [classnum_toclasslabel[k] for k in y]
print('Confusion Matrix')

cm = confusion_matrix(test_labels, predicted_test_labels, labels=list(set(test_labels)))

df = pd.DataFrame(cm, columns=list(set(test_labels)))

plt.figure(figsize=(80,80))

sn.heatmap(df, annot=True)
df_predictions_compare = pd.DataFrame({'actual': test_labels, 'pred': predicted_test_labels})

df_predictions_compare['count'] = 1





misclass_df = df_predictions_compare[df_predictions_compare['actual'] != df_predictions_compare['pred']].groupby(['actual', 'pred']).sum().sort_values(['count'], ascending=False).reset_index()

misclass_df['pair'] = misclass_df['actual'] + ' / ' + misclass_df['pred']

misclass_df = misclass_df[['pair', 'count']].take(range(50))

misclass_df.sort_values(['count']).plot.barh(figsize=(8, 10), x='pair')

plt.title('Top misclassed pairs (actual / predicted)')
misclassed_pairs_array = misclass_df['pair'].str.split('/')

import matplotlib.image as mpimg

#plt.imshow(mpimg.imread('MyImage.png'))
rows = 10

cols = 4

size = 25



misclassed_indice = 0



fig,ax=plt.subplots(rows,cols)

fig.set_size_inches(size,size)



i = 0

j = 0

plt.rcParams["axes.grid"] = False

for misclassed_pair in misclassed_pairs_array:

    if (i < rows):

        actual_indices = [ind for ind,x in enumerate(test_labels) if x == misclassed_pair[0].strip()] 

        predicted_indices = [ind for ind,x in enumerate(test_labels) if x == misclassed_pair[1].strip()] 



        misclassed_pair0 = misclassed_pair[0]

        misclassed_pair1 = misclassed_pair[1]



        rand_indices_actual = np.random.default_rng().choice(len(actual_indices), size=2, replace=False)

        rand_indices_predicted = np.random.default_rng().choice(len(predicted_indices), size=2, replace=False)

        

        plt.grid(None)

        ax[i,0].imshow(mpimg.imread(test_generator.filepaths[actual_indices[rand_indices_actual[0]]]))

        plt.grid(None)

        ax[i,1].imshow(mpimg.imread(test_generator.filepaths[actual_indices[rand_indices_actual[1]]]))

        plt.grid(None)

        ax[i,2].imshow(mpimg.imread(test_generator.filepaths[predicted_indices[rand_indices_predicted[0]]]))

        plt.grid(None)

        ax[i,3].imshow(mpimg.imread(test_generator.filepaths[predicted_indices[rand_indices_predicted[1]]]))

        plt.grid(None)





        ax[i,0].set_title(f'Sample of: {misclassed_pair0}')

        ax[i,1].set_title(f'Sample of: {misclassed_pair0}')

        ax[i,2].set_title(f'Sample of: {misclassed_pair1}')

        ax[i,3].set_title(f'Sample of: {misclassed_pair1}')



        i += 1

        misclassed_indice += 1

        

plt.tight_layout()
name = '../input/stanford-dogs-dataset/images/Images/n02106166-Border_collie/n02106166_1032.jpg'

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(name).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])
name = '../input/stanford-dogs-dataset/images/Images/n02099712-Labrador_retriever/n02099712_1383.jpg'

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(name).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])
name = '../input/stanford-dogs-dataset/images/Images/n02104029-kuvasz/n02104029_1206.jpg'

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(name).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])
name = '../input/stanford-dogs-dataset/images/Images/n02106550-Rottweiler/n02106550_10222.jpg'

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(name).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])
name = '../input/stanford-dogs-dataset/images/Images/n02097658-silky_terrier/n02097658_10020.jpg'

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(name).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])

name = '../input/stanford-dogs-dataset/images/Images/n02111889-Samoyed/n02111889_1363.jpg'

print(name)

plt.imshow(mpimg.imread(name))



img = Image.open(name).resize((targetx, targety))

probabilities = new_model.predict(preprocess_input(np.expand_dims(img, axis=0)))

breed_list = tuple(zip(test_generator.class_indices.values(), test_generator.class_indices.keys()))



for i in probabilities[0].argsort()[-5:][::-1]: 

    print(probabilities[0][i], "  :  " , breed_list[i])




