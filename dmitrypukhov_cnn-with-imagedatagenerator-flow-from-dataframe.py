#####################################
# Libraries
#####################################
# Common libs
import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path

# Image processing
import imageio
import cv2
import skimage.transform
#from skimage.transform import rescale, resize, downscale_local_mean

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

# ML, statistics
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

# Tensorflow
#from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

#############################################
# Settings
#############################################
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# Settings
plt.style.use('fivethirtyeight')
#plt.style.use('seaborn')

# toy=True - development mode, small samples, limited training, fast run
# toy=False - full data, slow learning and run
toy=False

all_df = pd.read_csv("../input/train_labels.csv")
if toy:
    all_df = all_df.sample(50000)

all_df.head()
all_df.label.value_counts().plot(kind='bar')
plt.title('Labels counts')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
class ImageViewer:
    def read_img(self,id, folder='train'):
        """
        Read image by it's id
        """
        file='../input/' + folder + '/' + str(id) + '.tif'
        im=cv2.imread(file)
        return im

    def draw_sample_images(self):
        """
        Draw cancer and healthy images for EDA
        """
        ncols=4
        f, ax = plt.subplots(nrows=2,ncols=ncols, 
                             figsize=(4*ncols,5*2))
        i=-1
        captions=['Pathology', 'Good']
        # Draw one row for cancer, one row for healthy images
        for label in [0,1]:
            i=i+1
            samples = all_df[all_df['label']==label]['id'].sample(ncols).values
            for j in range(0,ncols):
                file_id=samples[j]
                im=self.read_img(file_id)
                ax[i, j].imshow(im)
                ax[i, j].set_title(captions[i], fontsize=16)  
        plt.tight_layout()
        plt.show()
    
ImageViewer().draw_sample_images()
class DataPreparation:
    """
    Train/test
    """
    def train_test_split(self, all_df):
        """
        Balanced split to train, test and val
        """
        # Split to train and test before balancing
        train_df, test_df = train_test_split(all_df, random_state=24)
#         # Split train to train and validation datasets
#         train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=24)
        # Number of samples in each category
        ncat_bal = train_df['label'].value_counts().max()
        #ncat_bal = int(len(train_df)/train_df['label'].cat.categories.size)
        train_df = train_df.groupby('label', as_index=False).apply(lambda g:  g.sample(ncat_bal, replace=True, random_state=24)).reset_index(drop=True)
        return train_df, test_df
    
    def plot_balanced(self, train_df, all_df):
        """
        Plot samples per category before and after balancing
        """
        f, axs = plt.subplots(1,2,figsize=(12,4))
        # Before balancing
        all_df.label.value_counts().plot(kind='bar', ax=axs[0])
        axs[0].set_title('All labels')
        axs[0].set_xlabel('Label')
        axs[0].set_ylabel('Count')
        # After balancing
        train_df.label.value_counts().plot(kind='bar', ax=axs[1])
        axs[1].set_title('Train labels after balancing')
        axs[1].set_xlabel('Label')
        axs[1].set_ylabel('Count')
        plt.tight_layout()
        plt.show()


# Train/test/validation split with balanced labels in train
data_prep = DataPreparation()
train_df, test_df = data_prep.train_test_split(all_df)

# Plot before and after balancing
data_prep.plot_balanced(train_df, all_df)

class Generators:
    """
    Train, validation and test generators
    """
    def __init__(self, train_df, test_df):
        self.batch_size=32
        self.img_size=(64,64)
        
        # Base train/validation generator
        _datagen = ImageDataGenerator(
            rescale=1./255.,
            validation_split=0.25,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
            )
        # Train generator
        self.train_generator = _datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="../input/train/",
            x_col="id",
            y_col="label",
            has_ext=False,
            subset="training",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=self.img_size)
        print('Train generator created')
        # Validation generator
        self.val_generator = _datagen.flow_from_dataframe(
            dataframe=train_df,
            directory="../input/train/",
            x_col="id",
            y_col="label",
            has_ext=False,
            subset="validation",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="categorical",
            target_size=self.img_size)    
        print('Validation generator created')
        # Test generator
        _test_datagen=ImageDataGenerator(rescale=1./255.)
        self.test_generator = _test_datagen.flow_from_dataframe(
            dataframe=test_df,
            directory="../input/train/",
            x_col="id",
            y_col='label',
            has_ext=False,
            class_mode="categorical",
            batch_size=self.batch_size,
            seed=42,
            shuffle=False,
            target_size=self.img_size)     
        print('Test generator created')

        
# Create generators        
generators = Generators(train_df, test_df)
print("Generators created")
class ModelTrainer:
    """
    Create and fit the model
    """
    
    def __init__(self, generators):
        self.generators = generators
        self.img_width = generators.img_size[0]
        self.img_height = generators.img_size[1]
        
    def create_model_small(self):
        """
        Build CNN model using img_width, img_height from fields.
        """
        model=Sequential()
        model.add(Conv2D(16, kernel_size=3, input_shape=(self.img_width, self.img_height,3), activation='relu', padding='same'))
        model.add(MaxPooling2D(2))
        model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(64, activation = "relu"))        
        # 1 y label
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_model(self,
                    kernel_size = (3,3),
                    pool_size= (2,2),
                    first_filters = 32,
                    second_filters = 64,
                    third_filters = 128,
                    first_dense=256,
                    second_dense=128,
                    dropout_conv = 0.3,
                    dropout_dense = 0.3):

        model = Sequential()
        # First conv filters
        model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding="same",
                         input_shape = (self.img_width, self.img_height,3)))
        model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
        model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
        model.add(MaxPooling2D(pool_size = pool_size)) 
        model.add(Dropout(dropout_conv))

        # Second conv filter
        model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
        model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
        model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
        model.add(MaxPooling2D(pool_size = pool_size))
        model.add(Dropout(dropout_conv))

        # Third conv filter
        model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
        model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
        model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
        model.add(MaxPooling2D(pool_size = pool_size))
        model.add(Dropout(dropout_conv))

        model.add(Flatten())
        
        # First dense
        model.add(Dense(first_dense, activation = "relu"))
        model.add(Dropout(dropout_dense))
        # Second dense
        model.add(Dense(second_dense, activation = "relu"))
        model.add(Dropout(dropout_dense))
        
        # Out layer
        model.add(Dense(2, activation = "softmax"))

        model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    
    def train(self, model, toy):
        """
        Train the model
        """
        if toy:
            epochs=3
            steps_per_epoch=20
            validation_steps=2
        else:
            epochs=100
            steps_per_epoch=100
            #steps_per_epoch=30
            #steps_per_epoch=10
            validation_steps=5
            
        # We'll stop training if no improvement after some epochs
        earlystopper = EarlyStopping(monitor='val_acc', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
        # Save the best model during the traning
        checkpointer = ModelCheckpoint('best_model1.h5'
                                        ,monitor='val_acc'
                                        ,verbose=1
                                        ,save_best_only=True
                                        ,save_weights_only=True)
        # Train
        training = model.fit_generator(generator=self.generators.train_generator
                                ,epochs=epochs
                                ,steps_per_epoch=steps_per_epoch
                                ,validation_data=self.generators.val_generator
                                ,validation_steps=validation_steps
                                ,callbacks=[earlystopper, checkpointer, reduce_lr])
        # Get the best saved weights
        model.load_weights('best_model1.h5')
        return training
    
# Create and train the model
trainer = ModelTrainer(generators)

# model = trainer.create_model(kernel_size = (3,3),
#                     pool_size= (2,2),
#                     first_filters = 64,
#                     second_filters = 128,
#                     third_filters = 256,
#                     first_dense=256,
#                     second_dense=128,
#                     dropout_conv = 0.4,
#                     dropout_dense = 0.3)

model = trainer.create_model(kernel_size = (3,3),
                    pool_size= (2,2),
                    first_filters = 128,
                    second_filters = 256,
                    third_filters = 512,
                    first_dense=256,
                    second_dense=128,
                    dropout_conv = 0.3,
                    dropout_dense = 0.2)

model.summary()
training=trainer.train(model, toy)
print("Trained")
class Evaluator:
    """
    Evaluaion :predict on test data (not submission data from test folder)
    and print reports, plot results etc.
    """
     
    def __init__(self, model, training, generator, y_true):
        self.training = training
        self.generator = generator
        # predict the data
        steps=5
        self.y_pred_raw = model.predict_generator(self.generator, steps=steps)
        self.y_pred = np.argmax(self.y_pred_raw, axis=1)
        self.y_true=y_true[:len(self.y_pred)]        
    
    """
    Accuracy, evaluation
    """
    def plot_history(self):
        """
        Plot training history
        """
        ## Trained model analysis and evaluation
        f, ax = plt.subplots(1,2, figsize=(12,3))
        ax[0].plot(self.training.history['loss'], label="Loss")
        ax[0].plot(self.training.history['val_loss'], label="Validation loss")
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Accuracy
        ax[1].plot(self.training.history['acc'], label="Accuracy")
        ax[1].plot(self.training.history['val_acc'], label="Validation accuracy")
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        plt.tight_layout()
        plt.show()
    
    def plot_roc(self):
        #y_pred_keras = model.predict_generator(test_gen, steps=len(df_val), verbose=1)
        # Calculate roc
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.y_true, self.y_pred)
        auc_keras = auc(fpr_keras, tpr_keras)
        
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        
    def print_report(self):
        """
        Predict and evaluate using ground truth from labels
        Test generator did not shuffle 
        and we can use true labels for comparison
        """
        #Print classification report
        print(metrics.classification_report(self.y_true, self.y_pred))

# Create evaluator instance
evaluator = Evaluator(model, training, generators.test_generator, test_df.label.values)

# Draw accuracy and loss charts
evaluator.plot_history()

# ROC curve
evaluator.plot_roc()

# Classification report
evaluator.print_report()

class Submitter:
    """
    Predict and submit
    """
    def __init__(self, model, img_size):
        self.model = model
        batch_size=1000
        print("Initializing submitter")
        #Submission generator
        # flow_from_directory for input/test didn't work for me, so quick fix is to use flow_from_dataframe with list of files
        # Load list of files from test folder into dataframe
        test_files_df=pd.DataFrame()
        test_files_df['file']=os.listdir('../input/test/')
        print("Loaded test files list")
        # Create generator in it
        self.generator=ImageDataGenerator(rescale=1./255.).flow_from_dataframe(
                    dataframe=test_files_df,
                    directory="../input/test/",
                    x_col="file",
                    y_col=None,
                    has_ext=True,
                    class_mode=None,
                    batch_size=batch_size,
                    seed=42,
                    shuffle=False,
                    target_size=img_size)    
        print('Submission generator created')        

    def predict_for_submit(self):
        """
        Predict submission test data and form dataframe to submit
        """
        print("Forming submission dataframe...")
        # Predict
        y_pred = self.model.predict_generator(self.generator)
        y_pred = np.argmax(y_pred, axis=1)
        # Create submission df
        submission_df = pd.DataFrame({
            'id':self.generator.filenames,
            'label':y_pred })
        # Filename is id, remove extension .tif
        submission_df['id'] = submission_df['id'].apply(lambda x: x.split('.')[0])
        print(f"Submission dataframe created. Rows:{len(submission_df.values)}")
        
        # Write to csv
        submission_df.to_csv('submission.csv', index=False)
        print("Submission completed: written submission.csv")
        return submission_df

if not toy:
    # Get dataframe for submission
    submitter = Submitter(model, generators.img_size)
    submission_df = submitter.predict_for_submit()     
    submission_df.head()
else:
    submission_df = pd.DataFrame()
    print("Do not submit in toy mode")
df_sizes = pd.DataFrame({'Data type':['Train', 'Test', 'Submission'], 
                      'Row count':[len(train_df.values), len(test_df.values), len(submission_df.values)]})

sns.barplot(x='Data type', y='Row count', data=df_sizes)
plt.title('Rows in data')
plt.show()
