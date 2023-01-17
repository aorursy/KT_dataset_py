import json
import math
import os
import scipy
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
from imblearn.over_sampling import SMOTE
import keras.backend as K
from keras.applications.densenet import DenseNet121
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import seaborn as sns
sns.set()
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')
epochs = 100
batch_size = 6
img_size=224
seed = 999
learning_rate = 0.00001
np.random.seed(seed)
tf.set_random_seed(seed)

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print ('Training data shape', train_df.shape)
print ('Testing data shape', test_df.shape)
train_df['diagnosis'].value_counts().sort_index().plot(kind="bar",figsize=(12,5))
plt.title("Distribution", weight='bold',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Label", fontsize=17)
plt.ylabel("Frequency", fontsize=17);
fig, ax = plt.subplots(1, 5, figsize=(30, 12))
for i in range(5):
    sample = train_df[train_df['diagnosis'] == i].sample(1) #.sample(frac=1) will extract single element
    image_name = sample['id_code'].item()
    X = cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{image_name}.png')
    X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
    ax[i].set_title(f"Image: {image_name}\n Label = {sample['diagnosis'].item()}",fontsize=25)
    ax[i].axis('off')
    ax[i].imshow(X)

def crop_image(img,tol=7):
    if img.ndim ==2:
        print('nadim =2')
        mask = img>tol
        print('nadim =2 mask =',mask)
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        print('nadim =3')
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        print('nadim =3 mask =',mask)
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        print('check shape =',check_shape)
        if (check_shape == 0): 
            print('check shape 0 returned ')
            return img 
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            print('img1')
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            print('img2')
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            print('img3')
            img = np.stack([img1,img2,img3],axis=-1)
            print('img shape=',img.shape)
        return img
    
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    return image

def preprocess_image(image_path, desired_size=224):
    # Add Lighting, to improve quality
    im = load_ben_color(image_path)
    return im

def under_sample_make_all_same(df, categories, max_per_category):
    df = pd.concat([df[df['diagnosis'] == c][:max_per_category] for c in categories])
    df = df.sample(n=(max_per_category)*len(categories), replace=False, random_state=20031976)
    df.index = np.arange(len(df))
    return df

def circle_crop(img):   
    height, width, depth = img.shape    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    cropped_img = np.zeros((height, width), np.uint8)
    cv2.circle(cropped_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=cropped_img)
    img = crop_image(img)
    return img 

print('Training dataset shape now:',train_df.shape)

train_df = train_df.drop(train_df[train_df['diagnosis'] == 0].sample(n=805, replace=False).index)

N = train_df.shape[0]
x_train = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate((train_df['id_code'])):
    x_train[i, :, :, :] = preprocess_image(f'../input/aptos2019-blindness-detection/train_images/{image_id}.png')

N = test_df.shape[0]
x_test = np.empty((N, 224, 224, 3), dtype=np.uint8)

for i, image_id in enumerate((test_df['id_code'])):
    x_test[i, :, :, :] = preprocess_image(f'../input/aptos2019-blindness-detection/test_images/{image_id}.png')
    
y_train = pd.get_dummies(train_df['diagnosis']).values
# View the sample pre-processed images here
fig=plt.figure(figsize=(20, 10))
plt.title('Sample Img')
plt.imshow(x_train[0])

circle_crop_img = circle_crop(x_train[1])
fig=plt.figure(figsize=(20, 10))
plt.title('Sample Img')
plt.imshow(circle_crop_img)
print("x_train.shape=",x_train.shape)
print("y_train.shape=",y_train.shape)
print("x_test.shape=",x_test.shape)

x_samples, y_samples = SMOTE(random_state=seed).fit_sample(x_train.reshape(x_train.shape[0], -1), train_df['diagnosis'].ravel())

print("x_samples shape=",x_samples.shape)
print("x_samples shape=",y_samples.shape)

x_train = x_samples.reshape(x_samples.shape[0], 224, 224, 3)
y_train = pd.get_dummies(y_samples).values

print("x_train shape=",x_train.shape)
print("y_train shape=",y_train.shape)
y_train_multilabel = np.empty(y_train.shape, dtype=y_train.dtype)
y_train_multilabel[:, 4] = y_train[:, 4]

for i in range(3, -1, -1):
    y_train_multilabel[:, i] = np.logical_or(y_train[:, i], y_train_multilabel[:, i+1])

print("Original y_train:", y_train.sum(axis=0))
print("Multilabel version:", y_train_multilabel.sum(axis=0))
x_sptrain, x_spval, y_sptrain, y_spval = train_test_split(x_train, y_train_multilabel,test_size=0.25,random_state=seed )
datagen= ImageDataGenerator(
                zoom_range=0.10,        # set range for random zoom
                fill_mode='constant',   # set mode for filling points outside the input boundaries
                cval=0.,                # value used for fill_mode = "constant"
                horizontal_flip=True,   # randomly flip images
                vertical_flip=True,     # randomly flip images
            )

# Using original generator
data_generator = datagen.flow(x_sptrain, y_sptrain, batch_size=batch_size, seed=seed)
print("Image data augmentated ...")
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2*(p*r) / (p+r+K.epsilon())

print("Evaluation metrics defined ...")

densenet = DenseNet121(
    weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',
    include_top=False,
    input_shape=(224,224,3)
)

model = Sequential()
model.add(densenet)
model.add(layers.Dropout(0.5))
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(lr=learning_rate),
    metrics=['accuracy',mean_pred, precision, recall, f1_score, fbeta_score, fmeasure]
)
model.summary()
# callback to keep track of kappa score during training
class metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []
        
    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1
        
        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"Epoch: {epoch+1} val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return
    
kappa_score = metrics()
early_stop = EarlyStopping(
                monitor='val_loss', #Quantity to be monitored.
                mode='auto', #direction is automatically inferred from the name of the monitored quantity
                verbose=1, #verbosity mode.
                patience=8 #Number of epochs with no improvement after which training will be stopped
              )


history = model.fit_generator(
    data_generator,
    steps_per_epoch=x_train.shape[0] / batch_size,
    epochs=epochs,
    validation_data=(x_spval, y_spval),
    callbacks=[kappa_score,early_stop],
    verbose=1
)    
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
history_df = pd.DataFrame(history.history)
history_df.head(20)
y_test = model.predict(x_test) > 0.5
y_test = y_test.astype(int).sum(axis=1) - 1

test_df['diagnosis'] = y_test
test_df.to_csv('submission.csv',index=False)
test_df.diagnosis.value_counts()
f, ax = plt.subplots(figsize=(14, 8.7))
ax = sns.countplot(x="diagnosis", data=test_df, palette="cool")
plt.show()
