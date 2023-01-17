!pip install utils
import numpy as np 
import pandas as pd 
from utils import *
from glob import glob
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from itertools import chain
from datetime import datetime
import statistics
from tqdm import tqdm
import tensorflow as tf

print('Import Complete')
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras import Sequential

from keras import backend as K
import keras 

from sklearn.metrics import roc_curve, auc, roc_auc_score
print("TensorFlow version: ", tf.__version__)
print("Num GPUs Used: ", len(tf.config.experimental.list_physical_devices('GPU')))
if not os.path.exists('logs'):
    os.makedirs('logs')
    
if not os.path.exists('callbacks'):
    os.makedirs('callbacks')
    
if not os.path.exists('training_1'):
    os.makedirs('training_1')
    
CALLBACKS_DIR = '/kaggle/working/callbacks/'

# Disease Names / Class Labels 
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
labels_train_val = pd.read_csv('/kaggle/input/data/train_val_list.txt')
labels_train_val.columns = ['Image_Index']

labels_test = pd.read_csv('/kaggle/input/data/test_list.txt')
labels_test.columns = ['Image_Index']
labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']
for diseases in tqdm(disease_labels): #TQDM is a progress bar setting
    labels_df[diseases] = labels_df['Finding_Labels'].map(lambda result: 1 if diseases in result else 0)
train_val_merge = pd.merge(left=labels_train_val, right=labels_df, left_on='Image_Index', right_on='Image_Index')

test_merge = pd.merge(left=labels_test, right=labels_df, left_on='Image_Index', right_on='Image_Index')
train_val_merge['Finding_Labels'] = train_val_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

test_merge['Finding_Labels'] = test_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])
num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}

train_val_merge['Paths'] = train_val_merge['Image_Index'].map(img_path.get)
test_merge['Paths'] = test_merge['Image_Index'].map(img_path.get)
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])
# train_df, val_df = train_test_split(patients, test_size = 0.0596, random_state = 2019, shuffle= True)  

print('No. of Unique Patients in Train dataset : ',len(patients))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(patients)]
print('Training Dataframe   : ', train_df.shape[0],' images')

val_df, test_df = train_test_split(test_patients, test_size = 0.58, random_state = 2019, shuffle= True) 

print('No. of Unique Patients in Validation dataset : ',len(val_df))
val_df = test_merge[test_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('No. of Unique Patients in Test dataset : ',len(test_df))
test_df = test_merge[test_merge['Patient_ID'].isin(test_df)]
print('Testing Dataframe   : ', test_df.shape[0],' images')


IMG_SIZE = (224, 224)
train_data_gen = ImageDataGenerator(rescale=1./255,
                              samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True
                             )

test_data_gen = ImageDataGenerator(rescale=1./255)
SEED = 2

BATCH_SIZE = 32

train_gen = train_data_gen.flow_from_dataframe(dataframe=train_df, 
                                                directory=None,
                                                shuffle= True,
                                                seed = SEED,
                                                x_col = 'Paths',
                                                y_col = disease_labels, 
                                                target_size = IMG_SIZE,
                                                class_mode='raw',
                                                classes = disease_labels,
                                                color_mode = 'rgb',
                                                batch_size = BATCH_SIZE)

val_gen = train_data_gen.flow_from_dataframe(dataframe=val_df, 
                                                directory=None,
                                                shuffle= True,
                                                seed = SEED,
                                                x_col = 'Paths',
                                                y_col = disease_labels, 
                                                target_size = IMG_SIZE,
                                                classes = disease_labels,
                                                class_mode='raw',
                                                color_mode = 'rgb',
                                                batch_size = BATCH_SIZE
                                                )
print(BATCH_SIZE)

IMG_SHAPE = (224,224,3)

EPOCHS =10
!pip install -q pyyaml h5py
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1,mode='min',cooldown=0,min_lr=1e-8)

checkpoint_path = 'weights.h5'
checkpoint_dir = os.path.dirname(checkpoint_path)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path,
                                                                save_weights_only=True,
                                                                monitor='val_acc',
                                                                mode='max',
                                                                save_best_only=True,
                                                                verbose = 1
                                                              )

OPTIMIZER = Adam(learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999)

LOSS = BinaryCrossentropy()
METRICS = ['BinaryAccuracy']
    
import keras.backend as kb
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import shutil
import warnings
import json

class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, generator, class_names, weights_path, stats=None):
        super(Callback, self).__init__()
        self.generator = generator
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )
        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calcula el promedio de las Curvas ROC y guarda el mejor grupo de pesos
        de acuerdo a esta metrica
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"Learning Rate actual: {self.stats['lr']}")

        """
        y_hat shape: (#ejemplos, len(etiquetas))
        y: [(#ejemplos, 1), (#ejemplos, 1) ... (#ejemplos, 1)]
        """
        y_hat = self.model.predict_generator(self.generator,steps=self.generator.n/self.generator.batch_size)
        y = self.generator.labels

        print(f"*** epoch#{epoch + 1} Curvas ROC Fase Entrenamiento ***")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print(f"{i+1}. {self.class_names[i]}: {score}")
        print("*********************************")

        mean_auroc = np.mean(current_auroc)
        print(f"Promedio Curvas ROC: {mean_auroc}")
        if mean_auroc > self.stats["best_mean_auroc"]:
            print(f"Actualización del resultado de las Curvas de ROC de: {self.stats['best_mean_auroc']} a {mean_auroc}")

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print(f"Actualización del archivo de logs: {self.best_auroc_log_path}")
            with open(self.best_auroc_log_path, "a") as f:
                f.write(f"(epoch#{epoch + 1}) auroc: {mean_auroc}, lr: {self.stats['lr']}\n")

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            print(f"Actualización del grupo de pesos: {self.weights_path} -> {self.best_weights_path}")
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************")
        return
training_stats = {}
auroc = MultipleClassAUROC(
    generator=val_gen,
    class_names=disease_labels,
    weights_path=checkpoint_path,
    stats=training_stats
)
with tf.device('/GPU:0'):
    
    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               pooling="avg")

    base_model.trainable = True

    x = base_model.output
    predictions = Dense(14, activation='sigmoid',name='PredLayer')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss = LOSS,
                  optimizer=OPTIMIZER,
                  metrics=METRICS
                 )
model.save_weights(checkpoint_path.format(epoch=0))
from keras.utils.vis_utils import model_to_dot
from IPython.display import Image
from keras.callbacks import TensorBoard

# Image(model_to_dot(model, show_shapes=True).create_png())
# STEPS_TRAIN = 80912 / 32 / 10
# STEPS_VAL = 5610 / 32 / 5

history = model.fit_generator(train_gen,
                            validation_data= val_gen,     
                            epochs=EPOCHS,
#                             batch_size = BATCH_SIZE,
                            verbose = 1,
                            shuffle = True,
                            callbacks=[TensorBoard(log_dir=os.path.join("logs")), reduce_lr, model_checkpoint_callback, auroc],
                            steps_per_epoch = train_gen.n/train_gen.batch_size, 
                            validation_steps = val_gen.n/val_gen.batch_size,
)
model.load_weights('weights.h5')
model.summary()
# print(history.history['binary_accuracy'])
history = model.fit_generator(train_gen,
                            validation_data= val_gen,     
                            epochs=EPOCHS-4,
#                             batch_size = BATCH_SIZE,
                            verbose = 1,
                            shuffle = True,
                            callbacks=[TensorBoard(log_dir=os.path.join("logs")), reduce_lr, model_checkpoint_callback, auroc],
                            steps_per_epoch = train_gen.n/train_gen.batch_size, 
                            validation_steps = val_gen.n/val_gen.batch_size,
)
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(EPOCHS)
plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 1)
plt.grid()
plt.plot(epochs_range, acc, label='Training Binary Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Binary Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Binary Accuracy', color='Green')
plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', color='red')
plt.show()
