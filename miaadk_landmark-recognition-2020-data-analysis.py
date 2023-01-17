from sklearn import utils
import numpy as np
import pandas as pd
pd.set_option('expand_frame_repr', False)

import seaborn as sb
import matplotlib.pyplot as plt
import cv2
import PIL

import os
import sys
from datetime import datetime
from tqdm import tqdm

!pip install -q efficientnet
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Activation
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import efficientnet.tfkeras as efn
Labels = pd.read_csv(r"../input/landmark-recognition-2020/train.csv")
Sampel_Submission = pd.read_csv(r"../input/landmark-recognition-2020/sample_submission.csv")

Labels = Labels.dropna()       # Dropping missing values
Labels = Labels.sample(frac=1).reset_index(drop=True)  

print("Sampel submission:\n", Sampel_Submission.head(), "\n")
print("Train labels:")
Labels.tail(10)
Classes = Labels["landmark_id"].to_numpy()
Num_Classes = len(np.unique(Classes))
plt.figure(figsize=(13, 8))
sb.distplot(Classes, kde=False, rug=True, color="teal")
plt.title("Histogram")
plt.xlabel("Class")
plt.ylabel("count")
plt.show()
plt.figure(figsize=(13, 8))
sb.kdeplot(Classes, color="mediumspringgreen", shade=True)
plt.title("Data distribution")
plt.xlabel("Class")
plt.show()
def get_image_path(image_name, data_path=None):      # data_path is main directory
    if data_path is None:
        data_path = r"../input/landmark-recognition-2020/train"
        
    image_name = image_name[0: 3].replace("", "/") + image_name + ".jpg"   # Example --> converts "abcdef"  to  "/a/b/c/abcdf.jpg"
    image_path = data_path + image_name
    return image_path


def show_images(paths, rows, columns):
    fig = plt.figure(figsize=(50, 50))
    
    for i in range(1, columns*rows +1):
        plt.xticks([]), plt.yticks([]), plt.zticks([])
        image = cv2.imread(paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, i)
        plt.imshow(image)
    plt.show()
Sizes = {}

for image_name in tqdm(Labels["id"].iloc[: int(len(Labels) / 10)]):
    image_path = get_image_path(image_name)
    image = PIL.Image.open(image_path)
    width, height = image.size
    image_size = (width, height)
    
    if str(image_size) in Sizes:        # if image size is in the dict
        Sizes[str(image_size)] += 1     # counter += 1     
    
    else:
        Sizes[str(image_size)] = 0
        Sizes[str(image_size)] += 1
# ===========================
Sizes
Categories = []
Count = list(Sizes.values())

for index in range(len(Sizes)):
    if Count[index] > 2000:
        Categories.append(list(Sizes)[index])
    
    else:
        Categories.append("")      
        

plt.figure(figsize=(13, 9))
plt.pie(Count, labels=Categories)
plt.title("Size distribution")
plt.show()
Rand_Indexes = np.random.randint(0, len(Labels), (500))
Rand_Image_Names = Labels["id"].iloc[Rand_Indexes]             # Random image names
Rand_Image_Names = Rand_Image_Names.reset_index(drop=True)     # Reseting index

Rand_Image_Paths = [get_image_path(image_name) for image_name in Rand_Image_Names]   # getting paths

show_images(Rand_Image_Paths, 8, 7)
Images_Path = r"../input/landmark-recognition-2020/{}"
Image_Size = (299, 299)
Validation_Split = 0.007

Batch_Size = 32
Epochs = 10

Classes = Labels["landmark_id"].to_numpy()
Num_Classes = len(np.unique(Classes))
Class_Weight = utils.class_weight.compute_class_weight('balanced',
                                                       np.unique(Classes),
                                                       Classes
                                                      )
Early_Stop = EarlyStopping(monitor="val_loss",
                           mode="auto",
                           verbose=1,
                           patience=1,
                           restore_best_weights=True
                          )
# Shuffeling
Labels = Labels.sample(frac=1)         # shuffeling labels
Labels = Labels.reset_index(drop=True) # Reseting index

# Convert to numpy array
Labels = Labels.to_numpy()

# Spiliting
Place = int(len(Labels) * Validation_Split)

Validation_Labels = Labels[0: Place]
Train_Labels = Labels[Place: ]

# Shuffeling
np.random.shuffle(Train_Labels)
np.random.shuffle(Validation_Labels)

# showing some info
print("Train Labels:\n", Train_Labels, "\n\n",
      "Validation Labels:\n", Validation_Labels)

print(f"\nValidation labels' length: {len(Validation_Labels)}\tTrain labels' length: {len(Train_Labels)}")
def data_generator(labels, data_path, image_size, num_classes, batch_size=32):
    images, targets = [], []
    
    while True:
        for label in labels:
            # getting image path
            image_name = label[0]
            image_path = image_name[0: 3].replace("", "/") + image_name + ".jpg"    # convert "abcdef"  to  "/a/b/c/abcdf.jpg"
            image_path = data_path + image_path                                     # converts  ""/a/b/c/abcdf.jpg" to ../input/landmark-recognition-2020/train/a/b/c/abcdf.jpg"
            
            # loading image
            image_string = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image_string, channels=3)
            
            target = label[1]
            
            # Preprocessing
            image = image / 255     # Normalization
            print(image)
            target = to_categorical(target, num_classes=num_classes)
            
            # appending to lists
            images.append(image)
            targets.append(target)
            
            if len(targets) >= batch_size:
                yield np.array(images, dtype='float32'), np.array(targets, dtype='float32')
                images, targets = [], []
Train_Gen = data_generator(
    Train_Labels,
    Images_Path.format("train"),
    Image_Size,
    Num_Classes,
    Batch_Size)

Validation_Gen = data_generator(
    Validation_Labels,
    Images_Path.format("train"),
    Image_Size,
    Num_Classes,
    Batch_Size)
#EfficientNetB0 = efn.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
#EfficientNetB1 = efn.EfficientNetB1(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
#EfficientNetB2 = efn.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
#EfficientNetB5 = efn.EfficientNetB5(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))

Resnet152V2 = keras.applications.ResNet152V2(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
Xception = keras.applications.Xception(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))

#VGG16 = keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
#VGG19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
#DenseNet201 = keras.applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
#InceptionV3 = keras.applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(*Image_Size, 3))
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
def make_model(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(Num_Classes, activation="softmax"))
    
    return model
Model = make_model(Xception)
Model.Name =f"Xception--epochs{Epochs}--batch size:{Batch_Size}"
TB_Callback = TensorBoard(log_dir=f"../input/output/{Model.Name}/", histogram_freq=1)  # TensorBoard

# Optimizers
SGD_Optimizer = SGD(lr=0.1)
Adam_Optimizer = Adam(lr=0.1)

# Model compiling
Model.compile(
    loss="binary_crossentropy",
    optimizer=SGD_Optimizer,
    metrics=["accuracy"]
)    

Model.summary()
    
# Model training
Model.fit(
    Train_Gen,
    steps_per_epoch=len(Train_Labels)/Batch_Size,
    verbose=1,
    epochs=Epochs,
    validation_data=Validation_Gen,
    validation_steps=len(Validation_Labels)/Batch_Size,
    class_weight=Class_Weight,
    callbacks=[Early_Stop, TB_Callback]
)
Model.save(f"{Model.Name}--{str(datetime.now())}")