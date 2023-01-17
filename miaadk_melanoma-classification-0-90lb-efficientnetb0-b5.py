!pip install -q efficientnet

import numpy as np
from numpy.random import shuffle
import pandas as pd 
import random
import matplotlib.pyplot as plt
import cv2
pd.set_option('expand_frame_repr', False)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, Concatenate, Activation, LeakyReLU, ReLU, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import efficientnet.tfkeras as efn

import time
import os
from tqdm import tqdm
import datetime
import copy
TB_Log_Dir = r"/kaggle/working/{}/"       # TensorBoard logs directory
Labels_Dir = r"../input/siim-isic-melanoma-classification/{}.csv"
Images_Dir = r"../input/siimisic-melanoma-classificationresized/Resized/{}"
Output_Dir = r"/kaggle/working/{}.h5"

Epochs = 10
Batch_Size = 32
Early_Stop = EarlyStopping(monitor="val_loss", mode="auto", verbose=1, patience=1, restore_best_weights=True)

Image_Size = (299, 299)
Class_Weight = {0: 1, 1: 3.35}

Train_Over_Sampel_Count = 20_000         # for data augmentation
Valid_Over_Sampel_Count = 3_254
def data_augmentation(labels, target_length):
    counter = 0
    while counter < target_length:
        for i in tqdm(range(len(labels))):
            if counter >= target_length:                # if we produced enough labels
                continue
                
            row = labels[i]
            target = row[-2]
            
            if target == 0:
                continue
            
            mode = random.randint(0, 6)
            row_copy = copy.deepcopy(row)
            row_copy[-1] = mode                         # row_copy["argu_mode"] = mode      
            labels = np.concatenate([labels, [row_copy]])
            
            counter += 1
            
    shuffle(labels)
    return labels
# Columns = ["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge","diagnosis", "benign_malignant", "target", "argu_mode"]

Labels = pd.read_csv(Labels_Dir.format("train"))
Labels = Labels.sample(frac=1).reset_index(drop=True)  

# adding new column
Labels["argu_mode"] = [None for i in range(len(Labels))]

# to_categorical
Labels["sex"].replace(["male"], 1, inplace=True)
Labels["sex"].replace(["female"], 0, inplace=True)

# geting positive cancer cases' indexs
Positive_Indexs = Labels.index[Labels['target'] == 1].tolist()     # Positive cases indexes
Negative_Indexs = Labels.index[Labels['target'] == 1].tolist()  

# Converting data frame to numpy array  
Labels = Labels.to_numpy()
Positive_Cases = Labels[tuple([Positive_Indexs])]
Negative_Cases = np.delete(Labels, (Positive_Indexs), axis=0)   

shuffle(Positive_Cases)
shuffle(Negative_Cases)

# splitting
Place = int(len(Negative_Cases) * 0.1)

Train_Labels = np.concatenate([Positive_Cases[:], Negative_Cases[Place:]])
Validation_Labels = np.concatenate([Positive_Cases[0: 4], Negative_Cases[:Place]])

# Data augmentation
Train_Labels = data_augmentation(Train_Labels, Train_Over_Sampel_Count)
#Validation_Labels = data_augmentation(Validation_Labels, Valid_Over_Sampel_Count)

# Shuffling data
shuffle(Train_Labels)
shuffle(Validation_Labels)

# Count positive cancer cases in dataset
Train_Positive_Count = np.count_nonzero([row[-2] for row in Train_Labels])            # Positive cases count in Train labels
Valid_Positive_Count = np.count_nonzero([row[-2] for row in Validation_Labels])       # Positive cases count in Test labels

# print datasets
print("Train Labels:\n", Train_Labels, "\n\n",
      "Validation Labels:\n", Validation_Labels,
      "\n\n", "\tTrain:", Train_Positive_Count,
      "\tValidation", Valid_Positive_Count)

print(f"\nLen Validation_Data: {len(Validation_Labels)}\tLen Train Data: {len(Train_Labels)}")
def data_generator(labels, imgs_dir, image_size, batch_size=32):
    images, targets = [], []
    
    while True:
        for index in range(len(labels)):
            Id, target, argu_mode = labels[index][0], labels[index][-2], labels[index][-1]
            
            image_path = f"{imgs_dir}/{Id}.jpg"
            image = cv2.imread(image_path, 1)
            #image = cv2.resize(image, image_size)
            
            if argu_mode == 0:                  # augmentation mode 
                pass
                
            if argu_mode == 1:
                image = cv2.flip(image, 1)      # horizintal
            
            elif argu_mode == 2:
                image = cv2.flip(image, 0)      # vertical
                matrix = np.float32([[1, 0, 20], [0, 1, 20]])
                image = cv2.warpAffine(image, matrix, image_size)          # shifting image
            
            elif argu_mode == 3:
                image = cv2.flip(image, -1)     # both   
                matrix = np.float32([[1, 0, 33], [0, 1, 33]])
                image = cv2.warpAffine(image, matrix, image_size)          # shifting image
            
            elif argu_mode == 4:                # rotating image clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                matrix = np.float32([[1, 0, 23], [0, 1, 25]])
                image = cv2.warpAffine(image, matrix, image_size)          # shifting image
            
            elif argu_mode == 5:                # rotating image counter clockwise
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            elif argu_mode == 6:
                image = image[25: image_size[1] - 25, 25: image_size[0] - 25]    # Cropping image
                image = cv2.resize(image, (image_size[0], image_size[1]))
                matrix = np.float32([[1, 0, 32], [0, 1, 32]])
                image = cv2.warpAffine(image, matrix, image_size)                # shifting image
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image / 255                # Normalizing image 
            image = np.float32(image)

            images.append(image)
            targets.append(target)
            
            if len(images) >= batch_size:
                yield np.array(images, dtype='float32'), np.array(targets, dtype='float32')
                images, targets = [], []
Train_Gen = data_generator(
    Train_Labels,
    Images_Dir.format("train"),
    Image_Size,
    Batch_Size)

Validation_Gen = data_generator(
    Validation_Labels,
    Images_Dir.format("train"),
    Image_Size,
    Batch_Size)
EfficientNetB1 = efn.EfficientNetB1(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
EfficientNetB2 = efn.EfficientNetB2(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
EfficientNetB5 = efn.EfficientNetB5(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
EfficientNetB0 = efn.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(Image_Size[0], Image_Size[1], 3))
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
    model.add(Dense(1, activation="sigmoid"))
    
    return model
Model = make_model(EfficientNetB1)
Model.Name =f"efficentB1"
TB_Callback = TensorBoard(log_dir=TB_Log_Dir.format(Model.Name), histogram_freq=1)

SGD_Optimizer = SGD(lr=0.1)
Adam_Optimizer = Adam(lr=0.1)
    
for layer in Model.layers:
    layer.trainable = True
    
Model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)    

Model.summary()
    
Model.fit(
    Train_Gen,
    steps_per_epoch=len(Train_Labels)/Batch_Size,
    verbose=1,
    epochs=Epochs,
    validation_data=Validation_Gen,
    validation_steps=len(Validation_Labels)/Batch_Size,
    class_weight=Class_Weight,
    callbacks=[lr_schedule, Early_Stop, TB_Callback]
)
Model.save(Output_Dir.format(f"{Model.Name}--{str(datetime.datetime.now())}"))
def predict(images, model, image_size=(299,299), is_rgb=True):
    predictions = []
    for image in images:
        image = cv2.resize(image, (299, 299))
                      
        if not is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                      
        image = np.float32(image)
        image = image / 255             # Normalizing imgae
        image = np.reshape(image, (1, 299, 299, 3))
        predictions.append(model.predict(image))
        
    return predictions
Sample_Submission = pd.read_csv(r"../input/siim-isic-melanoma-classification/sample_submission.csv")
Test_Labels = pd.read_csv(r"../input/siim-isic-melanoma-classification/test.csv")

print(Sample_Submission.tail())

My_Submission = {"image_name": [], "target": []}

for index in tqdm(range(len(Test_Labels))):
    image_name = Test_Labels["image_name"][index]
    
    image = cv2.imread(f"../input/siim-isic-melanoma-classification/jpeg/test/{image_name}.jpg")
    My_Submission["image_name"].append(image_name)
    My_Submission["target"].append(predict([image], Model, is_rgb=False)[0][0][0])
# =========

My_Submission = pd.DataFrame(My_Submission)
My_Submission.to_csv(r"submission.csv", index=False)
print("Sample submission:\n", Sample_Submission.head(), "\n\n", "My submission:\n", My_Submission.head(15), "\n", My_Submission.tail(15))