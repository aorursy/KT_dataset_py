import os
import cv2
import json
import math
import scipy
import numpy as np 
import pandas as pd 

from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from keras import layers
from keras import backend as K
from keras.applications import DenseNet121, DenseNet201, Xception, InceptionResNetV2
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, classification_report
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import KFold

from keras.models import Model, save_model,load_model
train_path = '/kaggle/input/super-ai-image-classification/train/train'
valid_path = '/kaggle/input/super-ai-image-classification/val/val' 
train_df   = pd.read_csv(f"{train_path}/train.csv")
### apply image path to every rows
train_df['id_path'] = train_df['id'].apply(lambda x : f"{train_path}/images/{x}")
train_df
# Show type of dataset
print(train_df.shape)
print(train_df['category'].value_counts())

train_df["category"].value_counts().plot(kind="pie")
# Draw Categories of Images
def draw_category_images(col_name,figure_cols, df, IMAGE_PATH):
    categories = (df.groupby([col_name])[col_name].nunique()).index
    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols, 
                         figsize=(4*figure_cols,4*len(categories))) # adjust size here
    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name]==cat].sample(figure_cols) # figure_cols is also the sample size
        for j in range(0,figure_cols):
            file=IMAGE_PATH + sample.iloc[j]['id']            
            im=cv2.imread(file)
            ax[i, j].imshow(im, resample=True, cmap='gray')
            ax[i, j].set_title(cat, fontsize=16)  
    plt.tight_layout()
    plt.show()

draw_category_images('category',4, train_df, f"{train_path}/images/")
### Configuration
IMG_SIZE    = 300
BATCH_SIZE  = 32
SEED        = 33
### resize image before feed to neural network
def preprocess_image(image_path, IMG_SIZE):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image
### check amount of training set
N = train_df.shape[0]
N
### Prepare 3x3 NPArray for preprocess images with Amount of training data (N)
x_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

### Add Preprocess data to x_train array
### 'tqdm' => progress bar
for i, image_path in enumerate(tqdm(train_df['id_path'])):
    x_train[i, :, :, :] = preprocess_image(image_path, IMG_SIZE)
### Image after resizing
x_train[0][0]
### Convert categorical variable into indicator variables.
y_train = pd.get_dummies(train_df['category']).values
y_train
### Setup Model
### compare between DenseNet , Xception , InceptionResnet
### readmore | https://www.kaggle.com/pytorch/densenet201
### readmore | review other keras neural network @ https://keras.io/api/applications/
def model_setup():
    ### densenet
    densenet = DenseNet201(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE,IMG_SIZE,3)
    )
    
    ### Xception
#     xception = Xception(
#         include_top=True,
#         weights="imagenet",
#         input_tensor=None,
#         input_shape=(IMG_SIZE,IMG_SIZE,3),
#         pooling=None,
#         classes=1000,
#         classifier_activation="softmax",
#     )
    
#     ### InceptionResNetV2
#     inception_resnet = InceptionResNetV2(
#         include_top=True,
#         weights="imagenet",
#         input_tensor=None,
#         input_shape=(IMG_SIZE,IMG_SIZE,3),
#         pooling=None,
#         classes=1000,
#         classifier_activation="softmax",
#         **kwargs
#     )

    base_model  = densenet #xception ,#inception_resnet ,#densenet
    GAP_layer   = layers.GlobalAveragePooling2D()
    drop_layer  = layers.Dropout(0.6)
    dense_layer = layers.Dense(2, activation='sigmoid', name='final_output')
    
    x = GAP_layer(base_model.layers[-1].output)
    x = drop_layer(x)
    final_output = dense_layer(x)
    model = Model(base_model.layers[0].input, final_output)
    
    return model
### Cross validation
acc_per_fold        = []
loss_per_fold       = []
kf = KFold(n_splits = 10)

# initial fold
fold_no = 1
learining_rate = 0.000005

# with pretrained weights, and layers we want
modelOne = model_setup() 
modelOne.compile(
    loss = 'binary_crossentropy',  # use binary_crossentropy for loss optimization
    optimizer = Adam(lr=learining_rate),
    metrics = ['accuracy']
)
### Training phase
### optimize for 10 loops with 10 epochs per loop
for train, test in kf.split(x_train, y_train):

    # Generate a print
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = modelOne.fit(x_train[train], y_train[train],
              batch_size=BATCH_SIZE,
              epochs=10,
              validation_data=(x_train[test], y_train[test]),
              verbose=1)
    
    # Generate evaluation metrics
    scores = modelOne.evaluate(x_train[test], y_train[test], verbose=0)
    
    print(f'Score for fold {fold_no}: {modelOne.metrics_names[0]} of {scores[0]}; {modelOne.metrics_names[1]} of {scores[1]*100}%')
    
    # append accuract and loss score
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1
# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
x_plot = range(len(loss_per_fold))

plt.plot(x_plot, loss_per_fold, label = "Train")

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Compared Loss')

plt.legend()
plt.show()
def load_valid_data(test_set_folder):
    list_test_data = []
    for filename in os.listdir(test_set_folder):
        fullpath = os.path.join(test_set_folder, filename)
        if os.path.isfile(fullpath):
            dic_test = {}
            dic_test['id_path'] = fullpath
            dic_test['id'] = filename
            list_test_data.append(dic_test)

    pdTestSet = pd.DataFrame(list_test_data)

    return pdTestSet
def process_valid_images(pdValidate):
    N = pdValidate.shape[0]
    x_val_set = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    for i, image_id in enumerate(tqdm(pdValidate['id_path'])):
        x_val_set[i, :, :, :] = preprocess_image(image_id, IMG_SIZE)
    
    return x_val_set
df_valid   = load_valid_data(f"{valid_path}/images")
x_validate = process_valid_images(df_valid)
### Prediction
predictions = modelOne.predict(x_validate).argmax(axis=-1)
predictions
### Prepare val.csv => submission file
df_valid['category'] = predictions
df_valid
### Export submission
df_valid[['id', 'category']].to_csv("val.csv", index=False)