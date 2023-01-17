
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import random
from imgaug import augmenters as iaa

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model,Model
from keras.layers import Activation,Dropout,Flatten,Dense,Input,BatchNormalization,Conv2D
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras

df = pd.read_csv("../input/train.csv")
print("Total number of unique ids:",df.Id.count())
print("Total number of images:", df.Id.count()*4)
df.head(2)
labels = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}
for key in labels.keys():
    df[labels[key]] = 0
    
def filltargets(row):
    tar = row.Target.split(" ")
    for i in tar:
        col = labels[int(i)]
        row[str(col)]=1
    return row

df = df.apply(filltargets, axis=1)
df = df.drop(["Target"],axis = 1)
df.head(2)
freq_df = df.drop(["Id"],axis=1).sum(axis=0).sort_values(ascending = True)
plt.figure(figsize=(15,15))
sns.barplot(y=freq_df.index.values, x=freq_df.values*100/len(df), order=freq_df.index,palette="Blues")

def correlated_distribution(label):
    df_ = df[df[label]==1]
    freq_df = df_.drop(["Id",label],axis=1).sum(axis=0).sort_values(ascending = True)
    plt.figure(figsize=(5,5))
    plt.xlabel("Distribution for "+label)
    sns.barplot(y=freq_df.index.values, x=freq_df.values*100/len(df_), order=freq_df.index,palette="Blues")

correlated_distribution("Endosomes")
correlated_distribution("Rods & rings")
correlated_distribution("Peroxisomes")
correlated_distribution("Cytosol")
correlated_distribution("Mitochondria")
freq_df = pd.DataFrame()
freq_df["number_of_targets"] = df.drop(["Id"],axis=1).sum(axis=1).sort_values(ascending = True)
count_perc = np.round(100 * freq_df["number_of_targets"].value_counts() / freq_df.shape[0], 2)
plt.figure(figsize=(20,5))
sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Blues")
plt.xlabel("Number of targets per image")
plt.ylabel("% of data")
shape = (299,299)

def load_image(id,path="../input/train/"):
    global shape
    R = np.array(Image.open(path+id+'_red.png'))
    G = np.array(Image.open(path+id+'_green.png'))
    B = np.array(Image.open(path+id+'_blue.png'))
    Y = np.array(Image.open(path+id+'_yellow.png'))

    image = np.stack((
        R/2+Y/2,
        G/2+Y/2, 
        B),-1)

    image = cv2.resize(image, (shape[0], shape[1]))
    image = np.divide(image, 255)
    return image  

plt.figure(figsize=(15,15))
for n in range(4):
    idx = np.random.randint(0,20000,1)
    im_name = df.Id[idx[0]]
    im = load_image(im_name)
    plt.subplot(1,4,n+1)
    plt.imshow(im)
plt.show()

path_to_train = '/kaggle/input/train/'
data = pd.read_csv('/kaggle/input/train.csv')

train_dataset_info = []
for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})
train_dataset_info = np.array(train_dataset_info)

from sklearn.model_selection import train_test_split
train_ids, test_ids, train_targets, test_target = train_test_split(
    data['Id'], data['Target'], test_size=0.2, random_state=42)
class data_generator:
    
    def create_train(dataset_info, batch_size, shape, augument=True):
        assert shape[2] == 3
        while True:
            random_indexes = np.random.choice(len(dataset_info), batch_size)
            batch_images = np.empty((batch_size, shape[0], shape[1], shape[2]))
            batch_labels = np.zeros((batch_size, 28))
            for i, idx in enumerate(random_indexes):
                image = data_generator.load_image(
                    dataset_info[idx]['path'], shape)   
                if augument:
                    image = data_generator.augment(image)
                batch_images[i] = image
                batch_labels[i][dataset_info[idx]['labels']] = 1
            yield batch_images, batch_labels
    
    def load_image(path, shape):
        R = np.array(Image.open(path+'_red.png'))
        G = np.array(Image.open(path+'_green.png'))
        B = np.array(Image.open(path+'_blue.png'))
        Y = np.array(Image.open(path+'_yellow.png'))

        image = np.stack((
            R/2+Y/2,
            G/2+Y/2, 
            B),-1)

        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image        
    
    def augment(image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=0),
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])], random_order=True)
        
        image_aug = augment_img.augment_image(image)
        return image_aug
train_datagen = data_generator.create_train(
    train_dataset_info, 5, (299,299,3), augument=True)

images, labels = next(train_datagen)

fig, ax = plt.subplots(1,5,figsize=(25,5))
for i in range(5):
    ax[i].imshow(images[i])
print('min: {0}, max: {1}'.format(images.min(), images.max()))
def create_model(input_shape, n_out):
    model = Sequential()
    model.add(InceptionResNetV2(include_top=False,input_shape= input_shape, pooling='avg', weights="imagenet"))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(n_out, activation='softmax'))
    return model
def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('acc')
    ax[2].plot(history.epoch, history.history["acc"], label="Train acc")
    ax[2].plot(history.epoch, history.history["val_acc"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
from keras.optimizers import SGD
model = create_model(
    input_shape=(299,299,3), 
    n_out=28)

checkpointer = ModelCheckpoint(
    '/kaggle/working/InceptionResNetV2.model',
    verbose=2, save_best_only=True)

BATCH_SIZE = 10
INPUT_SHAPE = (299,299,3)

train_generator = data_generator.create_train(
    train_dataset_info[train_ids.index], BATCH_SIZE, INPUT_SHAPE, augument=False)
validation_generator = data_generator.create_train(
    train_dataset_info[test_ids.index], 256, INPUT_SHAPE, augument=False)

model.layers[0].trainable = True

model.compile(
    loss='binary_crossentropy',  
    optimizer=SGD(lr = 0.0001, momentum=0.9, nesterov=True),
    metrics=['acc', f1])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=500,
    validation_data=next(validation_generator),
    epochs=15, 
    verbose=1,
    callbacks=[checkpointer])
show_history(history)
from tqdm import tqdm
submit = pd.read_csv('../input/sample_submission.csv')
predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = data_generator.load_image(path, INPUT_SHAPE)
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
    
submit['Predicted'] = predicted
submit.to_csv('submission.csv', index=False)