%reload_ext autoreload
%autoreload 2
%matplotlib inline
# Common libs
import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split


# Tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Settings
plt.style.use('fivethirtyeight')
#plt.style.use('seaborn')
data_dir = Path('../input/jovian-pytorch-z2g/Human protein atlas')
test_dir = data_dir/'test'
train_dir = data_dir/'train'
csv_dir = data_dir/'train.csv'
submission_dir = '../input/jovian-pytorch-z2g/submission.csv'
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold
    TP = tf.reduce_sum(tf.cast(prob&label, float), axis=0)
    TN = tf.reduce_sum(tf.cast((~prob)&(~label), float), axis=0)
    FP = tf.reduce_sum(tf.cast((prob)&(~label), float), axis=0)
    FN = tf.reduce_sum(tf.cast((~prob)&label, float), axis=0)
    precision = tf.reduce_mean(TP/(TP+FP+(1e-12)))
    recall = tf.reduce_mean(TP/(TP+FN+(1e-12)))
    F2 = (1 + beta**2) * precision *recall /((beta**2) *precision + recall + (1e-12))
    return tf.reduce_mean(F2)

labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}

indxes = {str(v):k for k,v in labels.items()}
batch_size = 32
seed = 2020
val_split = 0.2
img_size = (224,224,3)
nfolds = 5
seed_everything(seed)
data = pd.read_csv(csv_dir)
data.head()
# data['Label'] = data['Label'].apply((lambda x: x.split(" ")))
# from sklearn.preprocessing import MultiLabelBinarizer
# binarizer = MultiLabelBinarizer()
# df = pd.DataFrame(binarizer.fit_transform(data['Label']),columns=binarizer.classes_)
# data = pd.concat([data, df], axis=1)
# data
def fill_targets(row):
    row.Label = np.array(row.Label.split(" ")).astype(np.int)
    for num in row.Label:
        name = labels[int(num)]
        row.loc[name] = 1
    return row
for key in labels.keys():
    data[labels[key]] = 0
data = data.apply(fill_targets, axis=1)
data.head()
data['Image'] = data['Image'].apply(lambda x: str(x)+'.png')
data.head()
test_data = pd.read_csv(submission_dir)
test_data['Image'] = test_data['Image'].apply(lambda x: str(x)+'.png')
test_data.head()
y = list(data.columns[2:])
test_names = test_data.Image.values
test_labels = pd.DataFrame(data=test_names, columns=["Image"])
for col in data.columns.values:
    if col != "Image":
        test_labels[col] = 0
test_labels.head(1)
class Generators:
    """
    Train, validation and test generators
    """
    def __init__(self, train_df, test_df):
        self.batch_size=50
        self.img_size=(224,224)
        
        # Base train/validation generator
        train_datagen = ImageDataGenerator(
            rescale=1./255.,
            validation_split=0.20,
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=True
            )
        test_datagen = ImageDataGenerator(rescale = 1./255.)
        # Train generator
        self.train_generator = train_datagen.flow_from_dataframe(
            dataframe=data,
            directory=train_dir,
            x_col='Image',
            y_col=y,
            has_ext=True,
            subset="training",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="raw",
            target_size=self.img_size)
        print('Train generator created')
        # Validation generator
        self.val_generator = train_datagen.flow_from_dataframe(
            dataframe=data,
            directory=train_dir,
            x_col="Image",
            y_col=y,
            has_ext=True,
            subset="validation",
            batch_size=self.batch_size,
            seed=42,
            shuffle=True,
            class_mode="raw",
            target_size=self.img_size)    
        print('Validation generator created')
        #test generator 
        self.test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_labels,
            directory=test_dir,
            x_col='Image',
            y_col=y,
            has_ext=True,
            class_mode="raw",
            batch_size= batch_size,
            shuffle=False,
            target_size=(224,224))
        print('Test generator created')
         
# Create generators        
generators = Generators(data, test_data)
print("Generators created")
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(20,10))
  for n in range(9):
      ab =[]
      for i, label in enumerate(label_batch[n]):
            if label ==1:
                ab.append(labels[i])
      ax = plt.subplot(3,3,n+1)
      plt.imshow(image_batch[n])
      plt.title((ab))
      plt.axis('off')
#       plt.tight_layout()
image_batch, label_batch = next(generators.train_generator)
show_batch(image_batch, label_batch)
import tensorflow_hub as hub
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))
feature_extractor_layer.trainable = False
base_model = tf.keras.applications.ResNet50(input_shape=(224,224,3),
                                              include_top=False,
                                              weights='imagenet')
import tensorflow_addons as tfa
fbeta=tfa.metrics.FBetaScore(num_classes=10, average="micro", threshold = 0.5)
class ModelTrainer:
    """
    Create and fit the model
    """
    
    def __init__(self, generators):
        self.generators = generators
        self.img_width = generators.img_size[0]
        self.img_height = generators.img_size[1]
    
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

        model = Sequential(
#             [feature_extractor_layer,
#                           keras.layers.GlobalAveragePooling2D()]
        )
#         # First conv filters
#         model.add(Conv2D(first_filters, kernel_size, activation = 'relu', padding="same",
#                          input_shape = (self.img_width, self.img_height,3)))
#         model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
#         model.add(Conv2D(first_filters, kernel_size, padding="same", activation = 'relu'))
#         model.add(MaxPooling2D(pool_size = pool_size)) 
#         model.add(Dropout(dropout_conv))

#         # Second conv filter
#         model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
#         model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
#         model.add(Conv2D(second_filters, kernel_size, padding="same", activation ='relu'))
#         model.add(MaxPooling2D(pool_size = pool_size))
#         model.add(Dropout(dropout_conv))

#         # Third conv filter
#         model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
#         model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
#         model.add(Conv2D(third_filters, kernel_size, padding="same", activation ='relu'))
#         model.add(MaxPooling2D(pool_size = pool_size))
#         model.add(Dropout(dropout_conv))

#         model.add(Flatten())
        model.add(feature_extractor_layer)
        # First dense
        model.add(Dense(first_dense, activation = "relu"))
        model.add(Dropout(dropout_dense))
        
        # Second dense
        model.add(Dense(second_dense, activation = "relu"))
        model.add(Dropout(dropout_dense))
        
        # Out layer
        model.add(Dense(10, activation = "sigmoid"))

        model.compile(optimizer=Adam(lr=0.0001), 
                      loss='binary_crossentropy', metrics=[fbeta])
        return model
        
    
    def train(self, model):
        """
        Train the model
        """
    
        epochs=5
        steps_per_epoch=15389//50
        validation_steps=3847//50
            
        # We'll stop training if no improvement after some epochs
        earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                   verbose=1, mode='max', min_lr=0.00001)
        # Save the best model during the traning
        checkpointer = ModelCheckpoint("classfication.h5"
                                        ,monitor='val_loss'
                                        ,verbose=1
                                        ,save_best_only=True
                                        ,save_weights_only=True)
        # Train
        training = model.fit(self.generators.train_generator
                                ,epochs=epochs
                                ,steps_per_epoch=steps_per_epoch
                                ,validation_data=self.generators.val_generator
                                ,validation_steps=validation_steps
                                ,callbacks=[earlystopper, reduce_lr, checkpointer])
        # Get the best saved weights
#         model.load_weights('classfication.h5')
        return training
    
# Create and train the model
trainer = ModelTrainer(generators)

model = trainer.create_model(kernel_size = (3,3),
#                     pool_size= (2,2),
#                     first_filters = 128,
#                     second_filters = 256,
#                     third_filters = 512,
                    first_dense=1024,
                    second_dense=512,
                    dropout_conv = 0.3,
                    dropout_dense = 0.2)

model.summary()
training=trainer.train(model)
print("Trained")
training.history.keys()
def plot_history(training):
        """
        Plot training history
        """
        ## Trained model analysis and evaluation
        f, ax = plt.subplots(1,2, figsize=(12,3))
        ax[0].plot(training.history['loss'], label="Loss")
        ax[0].plot(training.history['val_loss'], label="Validation loss")
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Accuracy
        ax[1].plot(training.history['fbeta_score'], label="F_score")
        ax[1].plot(training.history['val_fbeta_score'], label="Val F_score")
        ax[1].set_title('F_score')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('F_score')
        ax[1].legend()
        plt.tight_layout()
        plt.show()
plot_history(training)
preds = model.predict(generators.test_generator)
print(len(preds))
print(preds[:3])
bool_preds = preds.round()
bool_preds[:3]
sub = pd.DataFrame(bool_preds, columns = y)
sub.head()
def transform_to_target(row):
    target_list = []
    for col in sub.columns:
        if row[col] == 1:
            target_list.append(str(indxes[col]))
    if len(target_list) == 0:
        return None
    return " ".join(target_list)
sub["Predicted"] = sub.apply(lambda l: transform_to_target(l), axis=1)
sub.head()
a = pd.Series(sub.Predicted)
a[:3]
b = pd.Series(generators.test_generator.filenames)
b[:3]
final_df = pd.concat([b, a], axis=1)
final_df.columns = ['Image', 'Label']
final_df['Label'] = final_df['Label'].apply(lambda x: str(x).strip('[').strip(']').strip(','))
final_df['Label'] = final_df['Label'].apply(lambda x: str(x).replace(',', ' '))
final_df['Image'] = final_df['Image'].apply(lambda x: x.strip('png'))
final_df.to_csv('final_csv', index=False) 
