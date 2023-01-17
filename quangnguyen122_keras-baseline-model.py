import os
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import tensorflow as tf
import tensorflow.keras as keras

from fastprogress import master_bar, progress_bar

from PIL import Image

# Enable Eager Execution
#tf.enable_eager_execution() # No need use this, already enable with tf version >=2.0
tf.executing_eagerly() 
print(os.listdir("../input/vietai-advance-retinal-disease-detection-2020"))
ROOT = "../input/vietai-advance-retinal-disease-detection-2020"
TRAIN_DIR = "../input/vietai-advance-retinal-disease-detection-2020/train/train"
TEST_DIR = "../input/vietai-advance-retinal-disease-detection-2020/test/test"
data = pd.read_csv(os.path.join(ROOT, 'train.csv'))
data.head()
for label in data.columns[1:]:
    print("Distribution of", label)
    print(data[label].value_counts())
LABELS = data.columns[1:]
def build_label(row):
    return ",".join([LABELS[idx] for idx, val in enumerate(row[1:]) if val == 1])
        
data.apply(lambda x: build_label(x), axis=1).value_counts()
LABELS
train_data, val_data = train_test_split(data, test_size=0.2, random_state=2020)
IMAGE_SIZE = 224                              # Image size (224x224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)
BATCH_SIZE = 64                             
LEARNING_RATE = 0.001
LEARNING_RATE_SCHEDULE_FACTOR = 0.1           # Parameter used for reducing learning rate
LEARNING_RATE_SCHEDULE_PATIENCE = 5           # Parameter used for reducing learning rate
MAX_EPOCHS = 100                              # Maximum number of training epochs
def preprocessing_image(image):
    """
    Preprocess image after resize and augment data with ImageDataGenerator
    
    Parameters
    ----------
    image: numpy tensor with rank 3
        image to preprocessing
    
    Returns
    -------
    numpy tensor with rank 3
    """
    # TODO: augment more here
    
    return image
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                             featurewise_center=True,
                                                             featurewise_std_normalization=True,
                                                             preprocessing_function=preprocessing_image)
def build_label_list(row):
    return [LABELS[idx] for idx, val in enumerate(row[1:]) if val == 1]
        
train_data["label"] = train_data.apply(lambda x: build_label_list(x), axis=1)
val_data["label"] = val_data.apply(lambda x: build_label_list(x), axis=1)
train_gen = train_datagen.flow_from_dataframe(dataframe=train_data, 
                                        directory=TRAIN_DIR, 
                                        x_col="filename", 
                                        y_col="label",
                                        class_mode="categorical",
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                        batch_size=BATCH_SIZE)
val_gen = train_datagen.flow_from_dataframe(dataframe=val_data, 
                                        directory=TRAIN_DIR, 
                                        x_col="filename", 
                                        y_col="label",
                                        class_mode="categorical",
                                        shuffle=False,
                                        target_size=(IMAGE_SIZE,IMAGE_SIZE), 
                                        batch_size=BATCH_SIZE)
base_model = keras.applications.ResNet50(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
                                    include_top=False,
                                    weights='imagenet')
base_model.trainable = True

model = keras.Sequential([
  base_model,
  keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(len(LABELS), activation='sigmoid')
])

# Print out model summary
model.summary()
import tensorflow.keras.backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
mcp = keras.callbacks.ModelCheckpoint("resnet50.h5", monitor="val_f1", save_best_only=True, save_weights_only=True, verbose=1,mode='max')
rlr = keras.callbacks.ReduceLROnPlateau(monitor='val_f1', factor=LEARNING_RATE_SCHEDULE_FACTOR, mode='max', patience=LEARNING_RATE_SCHEDULE_PATIENCE, min_lr=1e-8, verbose=1)
callbacks = [mcp, rlr]
device = '/gpu:0'

with tf.device(device):
    steps_per_epoch = train_gen.n // BATCH_SIZE
    validation_steps = val_gen.n // BATCH_SIZE
    
    model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE), loss='binary_crossentropy', metrics=[f1])

    # Huấn luyện
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=MAX_EPOCHS,
                                  verbose=1,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps,
                                  callbacks=callbacks)
test_df = pd.read_csv(os.path.join(ROOT, 'sample_submission.csv'))
test_df.head()
test_gen = train_datagen.flow_from_dataframe(dataframe=test_df,
                                             directory=TEST_DIR,
                                             x_col="filename",
                                             class_mode=None,
                                             shuffle=False,
                                             target_size=(IMAGE_SIZE,IMAGE_SIZE),
                                             batch_size=BATCH_SIZE)
!ls
model.load_weights("resnet50.h5")
pred = model.predict_generator(test_gen)
labels = (train_gen.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels
LABELS = list(LABELS)

def probs2label(probs):
    ''' Return real index following LABELS
    '''
    global LABELS, labels
    return " ".join([str(LABELS.index(labels[idx])) for idx, prob in enumerate(probs) if prob > 0.5])
#test_df['predicted'] = np.apply_along_axis(probs2label, 1, pred)
for idx, row in test_df.iterrows():
    test_df.loc[idx]['predicted'] = probs2label(pred[idx])
test_df.to_csv("submission.csv", index=False)
test_df.head()
pred
train_gen.class_indices
