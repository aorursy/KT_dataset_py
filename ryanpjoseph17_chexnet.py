# Necessary Dependencies
import numpy as np 
import pandas as pd 
!pip install utils
from utils import *
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl

import os
from sklearn.model_selection import train_test_split
import statistics
from tqdm import tqdm

# DenseNet Dependencies
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dense,Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from matplotlib import pyplot as plt

# Classification Metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score

print('Started')
# Test if GPU present
print("Num GPUs Used: ", len(tf.config.experimental.list_physical_devices('GPU')))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['figure.figsize'] = (12, 10)
# Hyperparameters


# Preprocessing 
# 224
IMG_IND = 224
IMG_SIZE = (IMG_IND, IMG_IND)
IMG_SHAPE = (IMG_IND,IMG_IND,3)

# Model
OPTIMIZER = Adam(learning_rate=0.1,
                 beta_1=0.9,
                 beta_2=0.999)

LOSS = 'binary_crossentropy'
METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='BinaryAccuracy')
]

# Training 
EPOCHS = 100
BATCH_SIZE = 64

# Callbacks 
MODEL_CHECKPOINT_PERIOD = 25
LEARNING_RATE_PATIENCE = 5

# Binary Disease
binary_disease = ['Effusion']
binary_disease_str = 'Effusion'

class_weight = {
    0: 1.,
    1: 2.0
}
# Establish Directories 

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
# Load Stanford Images Distribution Files - Predetermined by the Stanford ChexNet Team

labels_train_val = pd.read_csv('/kaggle/input/train-val-images/train_val_list.txt')
labels_train_val.columns = ['Image_Index']

labels_test = pd.read_csv('/kaggle/input/tests-image/test_list.txt')
labels_test.columns = ['Image_Index']
# NIH Dataset Labels CSV File 

labels_df = pd.read_csv('/kaggle/input/data/Data_Entry_2017.csv')

labels_df.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                  'Patient_Age', 'Patient_Gender', 'View_Position',
                  'Original_Image_Width', 'Original_Image_Height',
                  'Original_Image_Pixel_Spacing_X',
                  'Original_Image_Pixel_Spacing_Y', 'dfd']

# Print Example Labels DataFrame
#print(labels_df.head(3))
# Binary Class Mapping 
labels_df[binary_disease_str] = labels_df['Finding_Labels'].map(lambda x: binary_disease_str in x)

# Print Class Mapping
print(labels_df[binary_disease_str].head(3))
# Combines Stanford Image Distribution csv file with NIH Label CSV
train_val_merge = pd.merge(left=labels_train_val, right=labels_df, left_on='Image_Index', right_on='Image_Index')

test_merge = pd.merge(left=labels_test, right=labels_df, left_on='Image_Index', right_on='Image_Index')
# Splitting Finding Labels
train_val_merge['Finding_Labels'] = train_val_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])

test_merge['Finding_Labels'] = test_merge['Finding_Labels'].apply(lambda s: [l for l in str(s).split('|')])
# Mapping Images to the theirs paths 
num_glob = glob('/kaggle/input/data/*/images/*.png')
img_path = {os.path.basename(x): x for x in num_glob}

# Training + Validation Mapping
train_val_merge['Paths'] = train_val_merge['Image_Index'].map(img_path.get)
# Testing Mapping
test_merge['Paths'] = test_merge['Image_Index'].map(img_path.get)

# Print Paths 
#train_val_merge.head(3)
# No Overlap in patients between the Train and Validation Data Sets
patients = np.unique(train_val_merge['Patient_ID'])
test_patients = np.unique(test_merge['Patient_ID'])

print('Number of Patients Between Train-Val Overall: ', len(patients))
print('Number of Patients Between Test Overall: ', len(test_patients))
# Train-Validation Split 
train_df, val_df = train_test_split(patients,
                                   test_size = 0.0669,
                                   random_state = 2019,
                                    shuffle= True
                                   )  


print('No. of Unique Patients in Train dataset : ',len(train_df))
train_df = train_val_merge[train_val_merge['Patient_ID'].isin(train_df)]
print('Training Dataframe   : ', train_df.shape[0],' images')

print('\nNo. of Unique Patients in Validtion dataset : ',len(val_df))
val_df = train_val_merge[train_val_merge['Patient_ID'].isin(val_df)]
print('Validation Dataframe   : ', val_df.shape[0],' images')

print('\nNo. of Unique Patients in Testing dataset : ',len(test_patients))
test_df = test_merge[test_merge['Patient_ID'].isin(test_patients)]
print('Testing Dataframe   : ', test_df.shape[0],' images')
# Number of Cases in each dataframe - before Oversampling 

print(f'# of {binary_disease} Cases in Training\n', train_df[binary_disease_str].value_counts(), '\n')

print(f'# of {binary_disease} Cases in Validation\n', val_df[binary_disease_str].value_counts(), '\n')

print(f'# of {binary_disease} Cases in Testing\n', test_df[binary_disease_str].value_counts(), '\n')
# Oversampling Method 
positive_cases = np.sum(train_df[binary_disease_str]==True)//2
oversample_factor = 2 # maximum number of cases in negative group so it isn't super rare
# Over and Undersampling Specific Disease Classes
train_df = train_df.groupby(['Patient_Gender', binary_disease_str]).apply(lambda x: x.sample(min(oversample_factor*positive_cases, x.shape[0]), replace = False)).reset_index(drop = True)

positive_cases = np.sum(val_df[binary_disease_str]==True)//2
val_df = val_df.groupby(['Patient_Gender', binary_disease_str]).apply(lambda x: x.sample(min(oversample_factor*positive_cases, x.shape[0]), replace = False)).reset_index(drop = True)

positive_cases = np.sum(test_df[binary_disease_str]==True)//2
test_df = test_df.groupby(['Patient_Gender', binary_disease_str]).apply(lambda x: x.sample(min(oversample_factor*positive_cases, x.shape[0]), replace = False)).reset_index(drop = True)
# Number of Cases in each dataframe - AFTER OVERSAMPLING

print(f'# of {binary_disease} Cases in Training\n', train_df[binary_disease_str].value_counts(), '\n')

print(f'# of {binary_disease} Cases in Validation\n', val_df[binary_disease_str].value_counts(), '\n')

print(f'# of {binary_disease} Cases in Testing\n', test_df[binary_disease_str].value_counts(), '\n')
# Reducing the Entire Dataset Size - Training  

num = 8095 
print(num)

original_train_df = train_df
train_df = train_df.groupby([binary_disease_str]).apply(lambda x: x.sample(num, replace = True)
                                                      ).reset_index(drop = True)

print('New Data Size:', train_df.shape[0], 'Old Size:', original_train_df.shape[0])
print(f'# of {binary_disease} Cases in Training\n', train_df[binary_disease_str].value_counts(), '\n')
# Reducing the Entire Dataset Size - Validation

num = 564 
print(num)

original_val_df = val_df
val_df = val_df.groupby([binary_disease_str]).apply(lambda x: x.sample(num, replace = True)
                                                      ).reset_index(drop = True)

print('New Data Size:', val_df.shape[0], 'Old Size:', original_val_df.shape[0])
print(f'# of {binary_disease} Cases in Validation\n', val_df[binary_disease_str].value_counts(), '\n')
# Number of Cases in each dataframe - AFTER REDUCTION

print(f'# of {binary_disease} Cases in Training\n', train_df[binary_disease_str].value_counts(), '\n')

print(f'# of {binary_disease} Cases in Validation\n', val_df[binary_disease_str].value_counts(), '\n')

print(f'# of {binary_disease} Cases in Testing\n', test_df[binary_disease_str].value_counts(), '\n')
# Image Data Generator - Data Augmentation 
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    samplewise_center=True, 
                                    samplewise_std_normalization=True, 
                                    horizontal_flip = True,
                                    zoom_range=0.1, 
                                    height_shift_range=0.05, 
                                    width_shift_range=0.05,
                                    rotation_range=5
                                    )
# Flow From DataFrame - Keras Preprocessing Pipeline 
# Training Flow From DataFrame 

train_gen = train_data_gen.flow_from_dataframe(dataframe=train_df, 
                                                directory=None,
                                                shuffle= True,
                                                seed = 2,
                                                x_col = 'Paths',
                                                y_col = binary_disease, 
                                                target_size = IMG_SIZE,
                                                class_mode='raw',
                                                classes = disease_labels,
                                                color_mode = 'rgb',
                                                batch_size = BATCH_SIZE)
# Validation Flow From DataFrame 

val_gen = train_data_gen.flow_from_dataframe(
                                            dataframe=val_df, 
                                            directory=None,
                                            shuffle= True,
                                            seed = 2,
                                            x_col = 'Paths',
                                            y_col = binary_disease, 
                                            target_size = IMG_SIZE,
                                            classes = disease_labels,
                                            class_mode='raw',
                                            color_mode = 'rgb',
                                            batch_size = BATCH_SIZE
                                            )

# Splitting of Validation Generator
x_val, y_val = next(val_gen)

# Declare TensorFlow Datasets for more efficient training
train_data = tf.data.Dataset.from_generator(lambda: train_gen,
                                            output_types=(tf.float32, tf.int32),
                                           output_shapes=([None, IMG_IND, IMG_IND, 3], [None, 1]))
val_data = tf.data.Dataset.from_generator(lambda: val_gen,
                                          output_types=(tf.float32, tf.int32),
                                         output_shapes=([None, IMG_IND, IMG_IND, 3], [None, 1]))
def feed_data(dataset):
    """
    This prefetches all the data for the model
    
    Arguments:
        dataset = tf.keras.FlowFromDataFrame 
        
    Returns:
        dataset = (tf.Dataset) the prefetched dataset (smaller batches)
    """
    
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  
    
    return dataset
image_paths = np.array(train_df['Paths'])
#print(image_paths)
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

images_to_augment = []

for image_path in image_paths[:4]:
    image = load_img(image_path, target_size=(IMG_IND, IMG_IND))
    image = img_to_array(image)
    images_to_augment.append(image)
    
images_to_augment = np.array(images_to_augment)

images_augmented = next(train_data_gen.flow(x=images_to_augment,
                                batch_size=10,
                                shuffle=False))
from tensorflow.keras.preprocessing.image import array_to_img

fig, axes = plt.subplots(2, 2)

for i in range(2):
    axes[i, 0].imshow(array_to_img(images_to_augment[i]), 
                      #horizontal_flip = True,
                      interpolation='nearest')
    
    axes[i, 1].imshow(array_to_img(images_augmented[i]), 
                      interpolation='nearest')
    
    axes[i, 0].set_xticks([])
    axes[i, 1].set_xticks([])
    
    axes[i, 0].set_yticks([])
    axes[i, 1].set_yticks([])
    
    axes[i, 0].set_yticks([])
    axes[i, 1].set_yticks([])
    
    axes[i, 0].set_xticks([])
    axes[i, 1].set_xticks([])
    
    axes[i, 0].set_yticks([])
    axes[i, 1].set_yticks([])
    
    axes[i, 0].set_yticks([])
    axes[i, 1].set_yticks([])
    
columns = ['Base Image', 'Augmented Image']
for ax, column in zip(axes[0], columns):
    ax.set_title(column) 
    
plt.show()
# Section of Code written by brucechou1983 - https://github.com/brucechou1983/CheXNet-Keras
# I have no experience with class weighting, brucechou1983 provided a very thorough explanation of the topic with example code
CLASS_NAMES = disease_labels
def get_class_weights(total_counts, class_positive_counts, multiply):
    """
    Calculate class_weight used in training
    Arguments:
    total_counts - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    multiply - int, positve weighting multiply
    use_class_balancing - boolean 
    Returns:
    class_weight - dict of dict, ex: {"Effusion": { 0: 0.01, 1: 0.99 }, ... }
    """
    def get_single_class_weight(pos_counts, total_counts):
        denominator = (total_counts - pos_counts) * multiply + pos_counts
        return {
            0: pos_counts / denominator,
            1: (denominator - pos_counts) / denominator,
        }

    class_names = list(class_positive_counts.keys())
    label_counts = np.array(list(class_positive_counts.values()))
    class_weights = []
    for i, class_name in enumerate(class_names):
        class_weights.append(get_single_class_weight(label_counts[i], total_counts))

    return class_weights

def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset
    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes
    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    total_count = df.shape[0]
    labels = df[class_names].as_matrix()
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    #class_positive_counts = (class_names, positive_counts)


    return total_count, class_positive_counts

newfds = 'newfds'
train_counts, train_pos_counts = get_sample_counts(newfds, "/kaggle/input/newfds/train", disease_labels)
class_weights = get_class_weights(train_counts, train_pos_counts, multiply=1)

# Code Snippet written by https://github.com/brucechou1983/CheXNet-Keras/blob/master/callback.py 
from keras.callbacks import Callback
import keras.backend as kb
import shutil

# Empty List for future Visualizations
MEAN_AUROC = []
DISEASE_AUROC = []
DISEASE_ = []
LR_LOG = []

class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, sequence, class_names, weights_path, stats=None, workers=1):
        super(Callback, self).__init__()
        #self.steps=STEPS ############################
        self.sequence = sequence
        self.workers = workers
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
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """
        print("\n*********************************")
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print(f"current learning rate: {self.stats['lr']}")
        #LR_LOG.append(self.stats['lr'])

        y_hat = model.predict(self.sequence,verbose=1)
        
        pred_indices = np.argmax(y_hat,axis=1)

        y = y_val 
    
        print(f"*** epoch#{epoch + 1} dev auroc ***")
        current_auroc = []
        try:
            score = roc_auc_score(y, y_hat)
        except ValueError:
            score = 0

        current_auroc.append(score)
        EPOCH = epoch + 1 

        print("*********************************")

        mean_auroc = np.mean(current_auroc)
        MEAN_AUROC.append(mean_auroc)
        print(f"Effusion auroc: {mean_auroc}\n")
        
        print("*********************************")
# Saves weights every 5 Epochs

# Dynamic Checkpoint File Name 
checkpoint_path = "training_1/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Model Checkpointing Callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=checkpoint_path, 
                                                verbose=1, 
                                                save_weights_only=True,
                                                period=MODEL_CHECKPOINT_PERIOD)

# Dynamic Learning Rate
reduced_lr = tf.keras.callbacks.ReduceLROnPlateau(
                                                monitor='val_loss',
                                                factor=.05,
                                                patience=LEARNING_RATE_PATIENCE,
                                                verbose=1,
                                                mode='min',
                                                cooldown=0,
                                                min_lr=1e-6 
                                                )

# Custom Callback displaying AUROC score across all diseases every epoch
auroc = MultipleClassAUROC(
                            sequence = x_val,
                            class_names=binary_disease,
                            weights_path=CALLBACKS_DIR,
                            stats={},
                            workers=1,
                            )


# CSV Logger of Metrics and Loss
csv_logger = CSVLogger('training.log')
# Training/Validation Steps

train_steps = train_gen.samples // BATCH_SIZE
val_steps = val_gen.samples // BATCH_SIZE 


print('Training Steps: ', train_steps)
print('Validation Steps: ', val_steps)
# Initializing GPU 
with tf.device('/GPU:0'):

    # Using Pre-trained Model (DenseNet)
    base_model = tf.keras.applications.DenseNet121(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet',
                                               pooling="avg")

    base_model.trainable = False

    x = base_model.output
    
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Dropout(0.2)(x)

    predictions = Dense(1, activation='sigmoid',name='Final')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss = 'binary_crossentropy',    
                  optimizer=OPTIMIZER,
                  metrics=METRICS
                                 )
# Training Model
history = model.fit(
                    feed_data(train_data),
                    steps_per_epoch = train_steps, 
                    validation_data= (feed_data(val_data)),    
                    validation_steps = val_steps, 
                    epochs=EPOCHS,
                    #use_multiprocessing=True,
                    class_weight = class_weight,
                    callbacks=[reduced_lr, cp_callback, auroc]
)
# Save Model Weights at End of Training 
model.save_weights('Model_finished')
# Graph of Binary Accuracy 

acc = history.history['BinaryAccuracy']
val_acc = history.history['val_BinaryAccuracy']


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
fig.savefig('TrainingValidationAccuracy.png')
# Graph of Loss
plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 2)
plt.grid()

acc = history.history['loss']
val_acc = history.history['val_loss']
epochs_range = range(EPOCHS)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', color='red')
plt.show()
fig.savefig('TrainingValidationLoss.png')
# Graph of Mean AUROC - Validation Data

plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 2)
#plt.grid()
plt.plot(MEAN_AUROC, label='Validation MEAN AUC_ROC')
plt.legend(loc='upper right')
plt.title('Validation AUC_ROC', color='Black')

plt.axhline(y=0.8638, color='r', linestyle='--')

plt.show()
# Graph of Mean AUROC - Validation Data

plt.figure(figsize=(40, 10))
plt.subplot(1, 2, 2)
plt.grid()
plt.plot(MEAN_AUROC, label='Validation MEAN AUC_ROC')
plt.legend(loc='upper right')
plt.title('Validation AUC_ROC', color='Black')

plt.axhline(y=0.8638, color='r', linestyle='--')

plt.ylim([0.0, 0.99])
plt.show()
# Testing
# Image Data Generator 
test_data_gen = ImageDataGenerator(rescale=1./255)

# Test Data Flow From DataFrame
test_gen = test_data_gen.flow_from_dataframe(dataframe=test_df, 
                                                directory=None,
                                                shuffle = True,
                                                seed = 3,
                                                x_col = 'Paths',
                                                y_col = binary_disease_str, 
                                                target_size = IMG_SIZE, 
                                                classes = disease_labels,
                                                class_mode='raw',
                                                color_mode = 'rgb',
                                                batch_size = 640
                                                )

x_test, y_test = next(test_gen)
score = model.evaluate(x_test , y_test, verbose=1)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")
# Using Model to Predict 
pred_Y = model.predict(x_test,
                        steps=64,
                        verbose = True)
# Confusion Matrix
from sklearn.metrics import confusion_matrix

plt.matshow(confusion_matrix(y_test, pred_Y>0.5))
# Precision 
from sklearn.metrics import classification_report

print(classification_report(y_test, pred_Y>0.5, target_names = ['Healthy', 'Effusion']))
# Graph of True Positive/False Positive 

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_test, pred_Y)
fig, ax1 = plt.subplots(1,1, figsize = (5, 5), dpi = 250)
ax1.plot(fpr, tpr, 'b.-', label = 'Effusion (AUC:%2.2f)' % roc_auc_score(y_test, pred_Y))
ax1.plot(fpr, fpr, 'k-', label = 'Random Guessing')
ax1.legend(loc = 4)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate');
fig.savefig('ROC_IMAGE_Atel')
# Single AUCROC Binary Score

print('My ROC Score (Effusion) : ', roc_auc_score(y_test, pred_Y))
print('Stanford ROC Score (Effusion) : 0.8638')
# Notes

