import numpy as np

import pandas as pd

from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Model

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing import image

from keras.models import Sequential

from glob import glob

import matplotlib.pyplot as plt



from keras.optimizers import Adam, SGD, RMSprop

import tensorflow as tf

import cv2

import glob

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

from tensorflow.python.keras import backend as K

import plotly.graph_objects as go

import plotly.offline as py

autosize =False



from plotly.subplots import make_subplots

import plotly.graph_objects as go



%matplotlib inline
import pandas as pd
train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'

train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')

test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')

#sub  = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train.head()
# as per an ongoing discussion, there are some duplicate images in the training data, these images might adversely impact our model, 

# so, lets remove these images

dup = pd.read_csv("/kaggle/input/siim-list-of-duplicates/2020_Challenge_duplicates.csv")



drop_idx_list = []

for dup_image in dup.ISIC_id_paired:

    for idx,image in enumerate(train.image_name):

        if image == dup_image:

            drop_idx_list.append(idx)



print("no. of duplicates in training dataset:",len(drop_idx_list))



train.drop(drop_idx_list,inplace=True)



print("updated dimensions of the training dataset:",train.shape)
train.target.value_counts()
# function to draw bar plot

import matplotlib.pyplot as plt

def draw_bar_plot(category,length,xlabel,ylabel,title,sub):

    plt.subplot(2,2,sub)

    plt.bar(category, length)

    plt.legend()

    plt.xlabel(xlabel, fontsize=15)

    plt.ylabel(ylabel, fontsize=15)

    plt.title(title, fontsize=15)

    #plt.show()
# lets visualize the class distribution

plt.figure(figsize = (8,6))

plt.bar(["Melanoma","Normal"],[len(train[train.target==1]), len(train[train.target==0])],color = 'rg')
df_benign=train[train['target']==0].sample(2000)

df_malignant=train[train['target']==1]
print('Benign Cases')

benign=[]

df_b=df_benign.head(30)

df_b=df_b.reset_index()

for i in range(30):

    img=cv2.imread(str(train_dir + df_benign['image_name'].iloc[i]+'.jpg'))

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    benign.append(img)

f, ax = plt.subplots(5,6, figsize=(10,6))

for i, img in enumerate(benign):

        ax[i//6, i%6].imshow(img)

        ax[i//6, i%6].axis('off')

        

plt.show()
print('Malignant Cases')

m=[]

df_m=df_malignant.head(30)

df_m=df_m.reset_index()

for i in range(30):

    img=cv2.imread(str(train_dir + df_m['image_name'].iloc[i]+'.jpg'))

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    m.append(img)

f, ax = plt.subplots(5,6, figsize=(10,6))

for i, img in enumerate(m):

        ax[i//6, i%6].imshow(img)

        ax[i//6, i%6].axis('off')

        

plt.show()
train.info()
import plotly.express as px



fig = px.pie(train, train['target'],color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
train_nona=train.dropna()

#fig = px.treemap(train_nona, path=['sex', 'age_approx'], values='target', color='target')

#fig.show()
entire=train.append(test)

affected_areas=pd.value_counts(entire['anatom_site_general_challenge'])

fig = go.Figure(data=[go.Pie(labels=affected_areas.index, values=affected_areas.values, hole=.3)])

fig.update_traces(hoverinfo='label+percent', textinfo='value',textfont_size=15,

                  marker=dict(colors=['#11100b','#ff3560'], line=dict(color='#FFFFFF', width=2.5)))

fig.update_layout(

    title='AFFECTED AREAS')

py.iplot(fig)
df_malignant=df_malignant.dropna()
age_counts=pd.value_counts(df_malignant['age_approx'])

gender_counts=pd.value_counts(df_malignant['sex'])

anatom_site_counts=pd.value_counts(df_malignant['anatom_site_general_challenge'])



fig = make_subplots(

    rows=1, cols=3,

    specs=[[{"type": "xy"},{"type": "domain"}, {"type": "xy"}]])



fig.add_trace(go.Bar(y=age_counts.values, x=age_counts.index),row=1, col=1)





fig.add_trace(go.Pie(values=gender_counts.values, labels=gender_counts.index,marker=dict(colors=['#100b','#f00560'], line=dict(color='#FFFFFF', width=2.5))),

              row=1, col=2)



fig.add_trace(go.Scatter(x=anatom_site_counts.index, y=anatom_site_counts.values),

              row=1, col=3)



fig.update_layout(height=700, showlegend=False)



fig.update_xaxes(title_text="Age", row=1, col=1)

fig.update_xaxes(title_text="Site", row=1, col=3)



# Update yaxis properties

fig.update_yaxes(title_text="Count", row=1, col=1)

fig.update_yaxes(title_text="Count", row=1, col=3)



# Update title and height

fig.update_layout(title_text="MALIGNANT DATA wrt AGE, GENDER, SITE",height=600, width=1000)



fig.show()
agecounts=pd.value_counts(train['age_approx'])

fig = px.bar(train, x=agecounts.index, y=agecounts.values)

fig.update_layout(title_text='Age counts of the training data')

fig.show()
agecounts=pd.value_counts(test['age_approx'])

fig = px.bar(test, x=agecounts.index, y=agecounts.values)

fig.update_layout(title_text='Age counts of the testing data')

fig.show()
fig = make_subplots(

    rows=1, cols=2,

    specs=[[{"type": "domain"}, {"type": "domain"}]])



site_train_counts=pd.value_counts(train['anatom_site_general_challenge'])



fig.add_trace(go.Pie(values=site_train_counts.values, labels=site_train_counts.index,title_text='Melanoma regions for training dataset',marker=dict(colors=['#100b','#f00560'], line=dict(color='#FFFFFF', width=2.5))),

              row=1, col=1)



site_test_counts=pd.value_counts(test['anatom_site_general_challenge'])



fig.add_trace(go.Pie(values=site_test_counts.values, labels=site_test_counts.index,title_text='Melanoma regions for testing dataset',marker=dict(colors=['#100b','#f00560'], line=dict(color='#FFFFFF', width=2.5))),

              row=1, col=2)



fig.update_layout(height=700, showlegend=False)



fig.show()
# Since this is a huge dataset, we would take a sample of it for training purpose



df_0=train[train['target']==0].sample(2000)

df_1=train[train['target']==1]

train=pd.concat([df_0,df_1])

train=train.reset_index()
# update image names with the whole path

def append_ext(fn):

    return train_dir+fn+".jpg"

train["image_name"]=train["image_name"].apply(append_ext)



def append_ext(fn):

    return test_dir+fn+".jpg"

test["image_name"]=test["image_name"].apply(append_ext)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(train['image_name'],train['target'], test_size=0.2, random_state=1234)



train=pd.DataFrame(X_train)

train.columns=['image_name']

train['target']=y_train



validation=pd.DataFrame(X_val)

validation.columns=['image_name']

validation['target']=y_val
# resizing the images

IMG_DIM = (224, 224)



# load images

train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train.image_name]

train_imgs = np.array(train_imgs)



validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation.image_name]

validation_imgs = np.array(validation_imgs)



print('Train dataset shape:', train_imgs.shape, 

      '\tValidation dataset shape:', validation_imgs.shape)
# define parameters for model training

batch_size = 128

num_classes = 2

epochs = 30

input_shape = (224, 224, 3)
# focal loss

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
# we will use Adam optimizer

opt = Adam(lr=1e-5)



#total number of iterations is always equal to the total number of training samples divided by the batch_size.

nb_train_steps = train.shape[0]//batch_size

nb_val_steps=validation.shape[0]//batch_size



print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
# Pixel Normalization and Image Augmentation

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50,

                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 

                                   horizontal_flip=True, fill_mode='nearest')



# no need to create augmentation images for validation data, only rescaling the pixels

val_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow(train_imgs, y_train, batch_size=batch_size)

val_generator = val_datagen.flow(validation_imgs, y_val, batch_size=batch_size)
img_id = 100

generator_100 = train_datagen.flow(train_imgs[img_id:img_id+1], train.target[img_id:img_id+1],

                                   batch_size=1)

aug_img = [next(generator_100) for i in range(0,5)]

fig, ax = plt.subplots(1,5, figsize=(16, 6))

print('Labels:', [item[1][0] for item in aug_img])

l = [ax[i].imshow(aug_img[i][0][0]) for i in range(0,5)]
import gc

del train

gc.collect()
from keras.applications import vgg16

from keras.models import Model

import keras



vgg = vgg16.VGG16(include_top=False, weights='imagenet', 

                                     input_shape=input_shape)



output = vgg.layers[-1].output

output = keras.layers.Flatten()(output)

vgg_model = Model(vgg.input, output)



vgg_model.trainable = False

for layer in vgg_model.layers:

    layer.trainable = False

    

import pandas as pd

pd.set_option('max_colwidth', -1)

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
vgg_model.trainable = True



set_trainable = False

for layer in vgg_model.layers:

    if layer.name in ['block5_conv1', 'block4_conv1']:

        set_trainable = True

    if set_trainable:

        layer.trainable = True

    else:

        layer.trainable = False

        

layers = [(layer, layer.name, layer.trainable) for layer in vgg_model.layers]

pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])    
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer

from keras.models import Sequential

from keras import optimizers



model = Sequential()

model.add(vgg_model)

model.add(Dense(512, activation='relu', input_dim=input_shape))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))





model.compile(loss=focal_loss(), metrics=[tf.keras.metrics.AUC()],optimizer=opt)
#!pip install livelossplot

#from livelossplot import PlotLossesKeras
from keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='loss', patience=3, verbose=1)
#cb=[PlotLossesKeras()]

model.fit_generator(train_generator, steps_per_epoch=nb_train_steps, epochs=epochs,callbacks=[es],

                              validation_data=val_generator, validation_steps=nb_val_steps, 

                              verbose=1)

x_test = np.load('../input/siimisic-melanoma-resized-images/x_test_224.npy')

x_test = x_test.astype('float16')

test_imgs_scaled = x_test / 255

del x_test

gc.collect()
target=[]

i = 0

for img in test_imgs_scaled:

    img1=np.reshape(img,(1,224,224,3))

    prediction=model.predict(img1)

    i = i + 1

    print("predicted image no.",i)

    target.append(prediction[0][0])
# submission file

sub=pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")

sub['target']=target

#sub.to_csv('submission.csv', index=False)

sub.head()
#img_csv=sub.copy()

tab_csv=pd.read_csv('../input/image-and-tab-csv-files/submission_tab.csv')

#img_csv.head()
img_csv=pd.read_csv('../input/image-and-tab-csv-files/submission_img.csv')
sub=img_csv.copy()

sub['target']= (img_csv['target'] + tab_csv['target'])/2

#sub['target']= img_csv['target'] * 0.8 + tab_csv['target'] * 0.2

sub.head()
sub.to_csv('submission.csv',index=False)