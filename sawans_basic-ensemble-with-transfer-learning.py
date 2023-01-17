import numpy as np
import pandas as pd

import keras
from keras.preprocessing import image
from keras.utils import to_categorical

from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation

from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras import backend as K

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
# Directory Listings for train and test images

train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
# Reading the csv files

train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
# Exploring the content of train file
train.head()
# Comparing the number of records in both categories
train['target'].value_counts()
train_benign = train[train['target']==0].sample(3000)
train_malign = train[train['target']==1]
train_samples = pd.concat([train_benign,train_malign])
train_samples.reset_index()
# Training data
train_labels = []
train_images =[]

for i in range(train_samples.shape[0]):
    train_images.append(train_dir+train_samples['image_name'].iloc[i]+'.jpg')
    train_labels.append(train_samples['target'].iloc[i])

df_train = pd.DataFrame(train_images)
df_train.columns =['images']
df_train['target'] = train_labels
# Test data
test_images =[]
for i in range(test.shape[0]):
    test_images.append(test_dir+test['image_name'].iloc[i]+'.jpg')

df_test = pd.DataFrame(test_images)
df_test.columns = ['images']
# Splitting the train data further into train and validation sets
X_train, X_val, y_train,y_val = train_test_split(df_train['images'],df_train['target'],test_size=0.2,random_state=0)

train = pd.DataFrame(X_train)
train.columns = ['images']
train['target']=y_train

validation = pd.DataFrame(X_val)
validation.columns = ['images']
validation['target']=y_val
def get_predictions(model,sub_df):
    target=[]
    for path in df_test['images']:
        img=cv2.imread(str(path))
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)/255.
        img=np.reshape(img,(1,224,224,3))
        prediction=model.predict(img)
        target.append(prediction[0][0])
    
    sub_df['target']=target
    return sub_df
train_datagen = ImageDataGenerator(preprocess_input,rescale=1./255,rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True)

val_datagen = ImageDataGenerator(preprocess_input,rescale=1./255)

image_size = 224

train_generator = train_datagen.flow_from_dataframe(
                    train,
                    x_col='images',
                    y_col ='target',
                    target_size=(image_size,image_size),
                    batch_size=8,
                    shuffle=True,
                    class_mode='raw')

validation_generator = val_datagen.flow_from_dataframe(
                    validation,
                    x_col='images',
                    y_col ='target',
                    target_size=(image_size,image_size),
                    batch_size=8,
                    shuffle=False,
                    class_mode='raw')
def vgg16_model(num_classes=None):
    model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
    x = Flatten()(model.output)
    output = Dense(1,activation='sigmoid')(x)
    model = Model(model.input,output)
    
    return model
vgg_conv = vgg16_model(1)
def focal_loss(alpha=0.25,gamma=2.0):
    def focal_crossentropy(y_true, y_pred):
        bce = K.binary_crossentropy(y_true, y_pred)
        
        y_pred = K.clip(y_pred, K.epsilon(), 1.- K.epsilon())
        p_t = (y_true*y_pred) + ((1-y_true)*(1-y_pred))
        
        alpha_factor = 1
        modulating_factor = 1

        alpha_factor = y_true*alpha + ((1-alpha)*(1-y_true))
        modulating_factor = K.pow((1-p_t), gamma)

        # compute the final loss and return
        return K.mean(alpha_factor*modulating_factor*bce, axis=-1)
    return focal_crossentropy
# Defining the optimizer and compiling the model
opt = Adam(lr=1e-5)
vgg_conv.compile(loss=focal_loss(),optimizer=opt,metrics=[keras.metrics.AUC()])
# Denining the num of epochs, batch_size and steps for training and validation
nb_epochs = 2
batch_size=8
nb_train_steps = train.shape[0]//batch_size  # // rounds off the result of division
nb_validation_steps = validation.shape[0]//batch_size
print("Number of training and validation steps are {} and {}".format(nb_train_steps,nb_validation_steps))
# Fitting the model
vgg_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_steps)
# Getting the predictions for test data

sub_dfs = []

sub_vgg16 = submission.copy()
sub_vgg16 = get_predictions(vgg_conv,sub_vgg16)
sub_vgg16.head()
sub_vgg16.to_csv('submission_vgg16.csv',index=False)
df_vgg16 = pd.read_csv('../input/subvgg16csv/submission_vgg16.csv')
df_vgg16.head()
sub_dfs = []
sub_dfs.append(df_vgg16)
def resnet50_model(num_classes=None):
    model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3),pooling='avg')
    x = Flatten()(model.output)
    output = Dense(1,activation='sigmoid')(x)
    model = Model(model.input,output)
    
    return model
resnet_conv = resnet50_model(1)
# Denining the optimizer and compiling the model
opt = Adam(lr=1e-5)
resnet_conv.compile(loss=focal_loss(),optimizer=opt,metrics=[keras.metrics.AUC()])
# Denining the num of epochs, batch_size and steps for training and validation
nb_epochs = 2
batch_size=8
nb_train_steps = train.shape[0]//batch_size  # // rounds off the result of division
nb_validation_steps = validation.shape[0]//batch_size
print("Number of training and validation steps are {} and {}".format(nb_train_steps,nb_validation_steps))
# Fitting the model
resnet_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_steps)
# Getting the predictions for test data


sub_resnet = submission.copy()
sub_resnet = get_predictions(resnet_conv,sub_resnet)
sub_resnet.head()
sub_resnet.to_csv('submission_resnet.csv',index=False)
df_resnet = pd.read_csv('../input/subresnetcsv/submission_resnet.csv')
sub_dfs.append(df_resnet)
!pip install -q efficientnet
import efficientnet.tfkeras as efn
def efficientnet_model(num_classes=None):
    
    model = keras.Sequential([
        
            efn.EfficientNetB7(
            weights='imagenet',
            include_top=False,
            input_shape=(224,224,3)),
        keras.layers.GlobalAveragePooling2D(), #Works like flatten layer but adds pooling to it to optimize data before feeding to FC layer
        keras.layers.Dense(1024,activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512,activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256,activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1,activation='sigmoid')
        
    ])
    return model
efficientnet_conv = efficientnet_model(1)
# Defining the optimizer and compiling the model
opt = Adam(lr=1e-5)
efficientnet_conv.compile(loss=focal_loss(),optimizer=opt,metrics=[keras.metrics.AUC()])
# Defining the num of epochs, batch_size and steps for training and validation
nb_epochs = 2
batch_size=8
nb_train_steps = train.shape[0]//batch_size  # // rounds off the result of division
nb_validation_steps = validation.shape[0]//batch_size
print("Number of training and validation steps are {} and {}".format(nb_train_steps,nb_validation_steps))
# Fitting the model
efficientnet_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_steps)
# Getting the predictions for test data


sub_efficientnet = submission.copy()
sub_efficientnet = get_predictions(efficientnet_conv,sub_efficientnet)
sub_efficientnet.head()
sub_efficientnet.to_csv('submission_efficientnet.csv',index=False)
df_efficient = pd.read_csv('../input/subefficientnetcsv/submission_efficientnet.csv')
sub_dfs.append(df_efficient)
from scipy.stats import rankdata
#Ranking results from all models
for i in range(3) :
    sub_dfs[i]['target'] = rankdata(sub_dfs[i]['target'], method='min')
sub_dfs[0]['target'] = sub_dfs[0]['target']*0.8 + sub_dfs[1]['target']*0.1 + sub_dfs[2]['target']*0.1
sub_dfs[0].head()
sub_dfs[0].to_csv('submission_blending.csv' , index = False)