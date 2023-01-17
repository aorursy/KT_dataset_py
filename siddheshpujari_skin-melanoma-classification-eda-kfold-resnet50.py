import os

from os import listdir  #returns a list that gives the names of the entries in the directory
from os.path import isfile,join

import pandas as pd
import numpy as np
from numpy import math
import seaborn as sns
sns.set(style='darkgrid')
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.show()

#Plotly
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True,theme='pearl')

#To read a dicom image , we can use pydicom
import pydicom

#Disable warnings
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import ResNet50
from keras.models import Sequential, Model,load_model
from keras.layers import Flatten,Dense
DEVICE = 'GPU'

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#The simplest way to get a list of entries in a directory is to use os.listdir()
#Pass in the directory you need the entries
os.listdir('../input/siim-isic-melanoma-classification')
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')
train.head()
test.head()
train.shape
#First we create a list of missing values by each feature
missing = list(train.isna().sum())

#then we create a list of columns and their missing values as inner list to a separate list
lst= []
i=0
for col in train.columns:
    insert_lst = [col,missing[i]]
    lst.append(insert_lst)
    i+=1

#finally create a dataframe
missing_df = pd.DataFrame(data=lst,columns=['Column_Name','Missing_Values'])

fig = px.bar(missing_df,x='Missing_Values',y='Column_Name',orientation='h',
             text='Missing_Values',title='Missing values in train dataset')
fig.update_traces(textposition='outside')
fig.show()

#Same thing for test file
missing = list(test.isna().sum())

lst= []
i=0
for col in test.columns:
    insert_lst = [col,missing[i]]
    lst.append(insert_lst)
    i+=1

#finally create a dataframe
missing_df = pd.DataFrame(data=lst,columns=['Column_Name','Missing_Values'])

fig = px.bar(missing_df,x='Missing_Values',y='Column_Name',orientation='h',
             text='Missing_Values',title='Missing values in test dataset')
fig.update_traces(textposition='outside')
fig.show()
# We separate the non nan values and nan values in separate dataframe.

not_null_sex = train[train['sex'].notnull()].reset_index(drop=True)
nan_sex = train[train['sex'].isnull()].reset_index(drop=True)

not_null_sex.head()
fig = plt.figure(figsize=(15,6))

fig1 = sns.countplot(data=not_null_sex,hue='sex',x='anatom_site_general_challenge')
#Check the anatom site in missing values.

nan_sex['anatom_site_general_challenge'].unique()
#Compute missing value with mode of sex

train['sex'].fillna(train['sex'].mode()[0],inplace=True)
train['age_approx'].value_counts()
train['age_approx'].median()
#Compute missing values with median

train['age_approx'].fillna(train['age_approx'].median(),inplace=True)
train['age_approx'].isna().sum()
train['anatom_site_general_challenge'].value_counts()
train['anatom_site_general_challenge'].fillna('NK',inplace=True)
test['anatom_site_general_challenge'].fillna('NK',inplace=True)

print('Train : {}'.format(train.isna().sum().sum()))
print('Test : {}'.format(test.isna().sum().sum()))
fig=plt.figure(figsize=(15,8))

labels = 'Benign','Malignant'

benign = train[train['benign_malignant']=='benign']
malignant = train[train['benign_malignant']=='malignant']
sizes = [len(benign),len(malignant)]

colors= ['lightskyblue','red']
#Plot
plt.pie(sizes,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)

plt.axis('equal');

print("There are {} number of patients in our dataset.".format(train['patient_id'].nunique()))
print("And there are total {} dicom images in the same dataset.".format(train['image_name'].nunique()))
# We groupby patient id and see the number of images wrt to each patient

x = train.groupby(['patient_id'],as_index=False)['image_name'].count()
x.sort_values(by="image_name",ascending=False)
x = train.groupby(['sex'],as_index=False)['benign_malignant'].count()
x = x.set_index('sex')
x
sns.countplot(data=train,x='sex',hue='benign_malignant');
# In the test dataset

sns.countplot(data=test,x='sex');
def create_dist(df,title):
    fig = plt.figure(figsize=(15,6))

    x= df["age_approx"].value_counts(normalize=True).to_frame()
    x = x.reset_index()
    ax = sns.barplot(data=x,y='age_approx',x='index')
    ax.set(xlabel='Age', ylabel='Percentage')
    ax.set(title=title);
create_dist(train,"Age distribution in train dataset")
create_dist(test,"Age distribution in test dataset")
fig = plt.figure(figsize=(15,6))

ax = sns.countplot(data=train,x='age_approx',hue='benign_malignant');
ax.set(title='Age vs Target');
fig = px.histogram(train,y='anatom_site_general_challenge',height=500,width=800,color_discrete_sequence=['indianred'])
fig.show()

fig = px.histogram(train,x='anatom_site_general_challenge',color='benign_malignant',barmode='group',height=500,width=800)
fig.show()
fig = px.histogram(test,y='anatom_site_general_challenge',height=500,width=800,color_discrete_sequence=['indianred'],title='Similar case in test dataset too')
fig.show()
fig = px.histogram(train,y='diagnosis',height=500,width=800,color_discrete_sequence=['goldenrod'],title='Diagnoses skin lesions')
fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
fig = px.histogram(train,x='diagnosis',color='benign_malignant',barmode='group',height=500,width=800)
fig.show()
#Create a separate images folder
train_images_dir = '../input/siim-isic-melanoma-classification/train/'
train_images = listdir(train_images_dir)

test_images_dir = '../input/siim-isic-melanoma-classification/test/'
test_images = listdir(test_images_dir)
#Define a function to plot randomly sampled images using pydicom

def plot_images(df):
    fig = plt.figure(figsize=(15,6))

    for i in range(1,11):
        image = df['image_name'][i]
        ds = pydicom.dcmread(train_images_dir+image+'.dcm')
        fig.add_subplot(2,5,i)
        plt.imshow(ds.pixel_array)
    
#We sample 11 rows from train dataset
random = train.sample(n=11)
random = random.reset_index(drop=True)

#Plot the images
plot_images(random)
#Similary , we sample random benign images
random = train[train['benign_malignant']=='benign'].sample(n=11)
random = random.reset_index(drop=True)

plot_images(random)
#Similary , we sample random malignant images
random = train[train['benign_malignant']=='malignant'].sample(n=11)
random = random.reset_index(drop=True)

plot_images(random)
#define a function for plotting anatomy sites

def plot_anatomy(target,anatomy_site):
    anatomy = train[train['anatom_site_general_challenge']==anatomy_site]

    fig = plt.figure(figsize=(15,6))
    for i in range(0,4):
        image = anatomy[anatomy['benign_malignant']==target].reset_index(drop=True)['image_name'][i]
        ds = pydicom.dcmread(train_images_dir+image+'.dcm')
        fig.add_subplot(2,4,i+1)
        plt.imshow(ds.pixel_array)
        plt.title(target)
    plt.suptitle(anatomy_site)
plot_anatomy('benign','head/neck')
plot_anatomy('malignant','head/neck')
plot_anatomy('benign','upper extremity')
plot_anatomy('malignant','upper extremity')
plot_anatomy('benign','lower extremity')
plot_anatomy('malignant','lower extremity')
plot_anatomy('benign','torso')
plot_anatomy('malignant','torso')
plot_anatomy('benign','palms/soles')
plot_anatomy('malignant','palms/soles')
plot_anatomy('benign','oral/genital')
plot_anatomy('malignant','oral/genital')
def plot_diagnosis(skin_lesion):
    fig = plt.figure(figsize=(12,6))

    for i in range(0,6):
        image = train[train['diagnosis']==skin_lesion].reset_index(drop=True)['image_name'][i]
        ds = pydicom.dcmread(train_images_dir+image+'.dcm')
        fig.add_subplot(2,3,i+1)
        plt.imshow(ds.pixel_array)
    plt.suptitle(skin_lesion.upper())
plot_diagnosis('nevus')
plot_diagnosis('melanoma')
plot_diagnosis('seborrheic keratosis')
plot_diagnosis('lentigo NOS')
plot_diagnosis('lichenoid keratosis')
plot_diagnosis('solar lentigo')
fig = plt.figure(figsize=(10,6))

image = train[train['diagnosis']=='cafe-au-lait macule'].reset_index(drop=True)['image_name'][0]
ds = pydicom.dcmread(train_images_dir+image+'.dcm')
fig.add_subplot(1,2,1)
plt.imshow(ds.pixel_array)
plt.title('cafe-au-lait macule'.upper())

image = train[train['diagnosis']=='atypical melanocytic proliferation'].reset_index(drop=True)['image_name'][0]
ds = pydicom.dcmread(train_images_dir+image+'.dcm')
fig.add_subplot(1,2,2)
plt.imshow(ds.pixel_array)
plt.title('atypical melanocytic proliferation'.upper());

#Import
from sklearn.model_selection import RepeatedKFold

def load_data_kfold(k):
    #X = train['image_name']
    #y = train['target']
    
    #X_train,X_val = tts(train_x, test_size=0.2, random_state=1234)

    #y_train = np.array(y_train)
    train_x = train[['image_name','target']]
    train_x['image_name'] = train_x['image_name'].apply(lambda x: x + '.jpg')
    folds = list(RepeatedKFold(n_splits=k, n_repeats=1, random_state=0).split(train_x))
    
    return folds,train_x

k = 3
folds,train_x = load_data_kfold(k)


folds
'''METRICS = [
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
]'''

def get_model():
    model =ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

    for layer in model.layers:
        layer.trainable = False 

    x=Flatten()(model.output)
    output=Dense(1,activation='softmax')(x)

    model = Model(model.input,output)
    
    model.compile(
    'Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )
    
    return model
    
model = get_model()
model.summary()


class Config:
    BATCH_SIZE = 64
    EPOCHS = 10
    HEIGHT = 224
    WIDTH = 224
for j, (train_idx, val_idx) in enumerate(folds):
    
    print('\nFold ',j)
    print('///////////////////////////////////')
    X_train_cv = train_x.iloc[train_idx]
    #y_train_cv = y_train[train_idx]
    X_valid_cv = train_x.iloc[val_idx]
    #y_valid_cv= y_train[val_idx]
    
    #name_weights = "final_model_fold" + str(j) + "_weights.h5"
    #callbacks = get_callbacks(name_weights = name_weights, patience_lr=10)

    train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                         rotation_range=360,
                         horizontal_flip=True,
                         vertical_flip=True)
    
    train_generator=train_datagen.flow_from_dataframe(
        dataframe=X_train_cv,
        directory='../input/siim-isic-melanoma-classification/jpeg/train/',
        x_col="image_name",
        y_col="target",
        class_mode="raw",
        batch_size=Config.BATCH_SIZE,
        target_size=(Config.HEIGHT, Config.WIDTH))

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    valid_generator=validation_datagen.flow_from_dataframe(
        dataframe=X_valid_cv,
        directory='../input/siim-isic-melanoma-classification/jpeg/train/',
        x_col="image_name",
        y_col="target",
        class_mode="raw", 
        batch_size=Config.BATCH_SIZE,   
        target_size=(Config.HEIGHT, Config.WIDTH))
    
    model = get_model()
    
    TRAINING_SIZE = len(train_generator)
    VALIDATION_SIZE = len(valid_generator)
    BATCH_SIZE = 64

    compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))
    steps_per_epoch = compute_steps_per_epoch(TRAINING_SIZE)
    validation_steps = compute_steps_per_epoch(VALIDATION_SIZE)
    
    history = model.fit_generator(generator=train_generator,
                                        steps_per_epoch=steps_per_epoch,
                                        validation_data=valid_generator,
                                        validation_steps=validation_steps,
                                        epochs=10,
                                        verbose=1)
    
    #print(model.evaluate(X_valid_cv['image_name'], X_valid_cv['target']))


test_x = test[['image_name']]

test_x['image_name'] = test_x['image_name'].apply(lambda x: x + '.jpg')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_dataframe(  
        dataframe=test_x,
        directory = '../input/siim-isic-melanoma-classification/jpeg/test/',
        x_col="image_name",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        target_size=(Config.HEIGHT, Config.WIDTH),
        seed=0)



preds = model.predict_generator(test_generator,verbose=1)
predicted_class_indices = np.argmax(preds, axis = 1)
predicted_class_indices
len(preds)
len(predicted_class_indices)
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub
sub['target'] = predicted_class_indices
sub.to_csv('submission.csv', index=False)