import tensorflow
!pip install livelossplot
!pip install tornado==4.5.3
import os

import pandas as pd

import xgboost as xgb

import numpy as np

from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, accuracy_score, f1_score



from sklearn.impute import KNNImputer

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc





import tensorflow

import cv2

import PIL

from IPython.display import Image, display

from keras.applications.vgg16 import VGG16,preprocess_input



import plotly.graph_objs as go

import plotly.graph_objects as go

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model, load_model

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, BatchNormalization, Activation

from keras.layers import GlobalMaxPooling2D

from keras.models import Model

from keras.optimizers import Adam, SGD, RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import gc

import skimage.io

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.python.keras import backend as K

from livelossplot import PlotLossesKeras





def clean_dataset(df):

    """

    Cleans data frame from NaNs, Infs and missing cells.

    """

    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    df.dropna(inplace=True)

    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)
TEST_CSV_PATH = '../input/siim-isic-melanoma-classification/test.csv'

TRAIN_CSV_PATH = '../input/siim-isic-melanoma-classification/train.csv'

TEST_JPEG_PATH = '../input/siim-isic-melanoma-classification/jpeg/test/'

TRAIN_JPEG_PATH = '../input/siim-isic-melanoma-classification/jpeg/train/'



train=pd.read_csv(TRAIN_CSV_PATH)

test=pd.read_csv(TEST_CSV_PATH)

train.head()

val1, val2 = train['target'].value_counts()

dist=train['target'].value_counts()

print(f"{(val1/(val1+val2))*100}% of benign data.")

print(f"{(val2/(val1+val2))*100}% of malign data.")
df_0=train[train['target']==0].sample(2000)

df_1=train[train['target']==1]

train=pd.concat([df_0,df_1])

train=train.reset_index()
labels=[]

data=[]

for i in range(train.shape[0]):

    data.append(TRAIN_JPEG_PATH + train['image_name'].iloc[i]+'.jpg')

    labels.append(train['target'].iloc[i])

df=pd.DataFrame(data)

df.columns=['images']

df['target']=labels



test_data=[]

for i in range(test.shape[0]):

    test_data.append(TEST_JPEG_PATH + test['image_name'].iloc[i]+'.jpg')

df_test=pd.DataFrame(test_data)

df_test.columns=['images']
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], test_size=0.2, random_state=1234)



train=pd.DataFrame(X_train)

train.columns=['images']

train['target']=y_train.astype(np.float32)



validation=pd.DataFrame(X_val)

validation.columns=['images']

validation['target']=y_val.astype(np.float32)
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(

    train,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    batch_size=8,

    shuffle=True,

    class_mode='raw')



validation_generator = val_datagen.flow_from_dataframe(

    validation,

    x_col='images',

    y_col='target',

    target_size=(224, 224),

    shuffle=False,

    batch_size=8,

    class_mode='raw')
def vgg16_model(num_classes=None):



    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x=Flatten()(model.output)

    output=Dense(1,activation='sigmoid')(x) # because we have to predict the AUC

    model=Model(model.input,output)

    

    return model



vgg_conv=vgg16_model(1)
def focal_loss(alpha=0.25, gamma=2.0):

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
opt = Adam(lr=1e-5)

vgg_conv.compile(loss=focal_loss(), metrics=[tf.keras.metrics.AUC()],optimizer=opt)
nb_epochs = 20

batch_size = 16

nb_train_steps = train.shape[0]//batch_size

nb_val_steps=validation.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_loss', verbose=1,

    save_best_only=True, mode='auto', period=1)
cb=[PlotLossesKeras(), checkpoint]

vgg_conv.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_steps,

    epochs=nb_epochs,

    validation_data=validation_generator,

    callbacks=cb,

    validation_steps=nb_val_steps)
# serialize model to JSON

model_json = vgg_conv.to_json()

with open("last_model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

vgg_conv.save_weights("last_model.h5")

print("Saved model to disk")
data_df = pd.read_csv(TRAIN_CSV_PATH) 

data_df = data_df.drop(columns=['benign_malignant', 'diagnosis', 'patient_id'])



# one-hot encoding

data_df = pd.concat([data_df,pd.get_dummies(data_df['sex'], prefix='sex', drop_first=True)],axis=1)

data_df = pd.concat([data_df,pd.get_dummies(data_df['anatom_site_general_challenge'], prefix='anatom', drop_first=True)],axis=1)

data_df.drop(['sex', 'anatom_site_general_challenge'],axis=1, inplace=True)



# normalization

column_names_to_normalize = ['age_approx']

x = data_df[column_names_to_normalize].values

scaler = MinMaxScaler() 

x_scaled = scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = data_df.index)

data_df[column_names_to_normalize] = df_temp



data_df.head()

loaded_vgg = load_model('./best_model.hdf5', custom_objects={'focal_crossentropy': focal_loss})



def vgg_prediction(image_name):

    #print(str(TRAIN_JPEG_PATH + image_name + '.jpg'))

    img = cv2.imread(str(TRAIN_JPEG_PATH + image_name + '.jpg'))

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img = np.reshape(img,(1,224,224,3))

    prediction = loaded_vgg.predict(img)

    return prediction[0][0]
data_df['vgg_prediction'] = data_df['image_name'].apply(vgg_prediction)

data_df
data_df.to_csv("train_with_vgg.csv")
y = data_df['target']

X_data = data_df.drop(columns=['target', 'image_name'])



# split the dataset into train and Test

seed = 7

test_size = 0.3

Xtrain, Xtest, ytrain, ytest = train_test_split(X_data, y, test_size=test_size, random_state=seed)



Xtrain.head()
data_dmmatrix= xgb.DMatrix(data=Xtrain,label=ytrain)



workers=4



param_bin = {

    'nthread':workers,

    'max_depth': 500,

    'eta': 0.01,

    'gamma':0,

    'subsample':0.8,

    'colsample_bytree':0.8,

    'objective': 'binary:logistic'}

epochs = 1000



model_bin = xgb.train(param_bin, data_dmmatrix, epochs)

xgb_test = xgb.DMatrix(Xtest, label=ytest)

predictions_bin = model_bin.predict(xgb_test)



auc_bin = roc_auc_score(ytest, predictions_bin)

print('binary:logistic ROC AUC=%.3f' % (auc_bin))



# calculate roc curves

fpr_bin, tpr_bin, _ = roc_curve(ytest, predictions_bin)



# plotting the roc curves of each xgboost objective model

pyplot.plot(fpr_bin, tpr_bin, linestyle='--', label='binary:logistic')

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

pyplot.legend()

pyplot.show()
def vgg_prediction_test(image_name):

    img = cv2.imread(str(TEST_JPEG_PATH + image_name + '.jpg'))

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img = np.reshape(img,(1,224,224,3))

    prediction = loaded_vgg.predict(img)

    return prediction[0][0]

 

eval_data_df = pd.read_csv(TEST_CSV_PATH)

imageNames_lst = eval_data_df['image_name']



eval_data_df = pd.concat([eval_data_df,pd.get_dummies(eval_data_df['sex'], prefix='sex', drop_first=True)],axis=1)

eval_data_df = pd.concat([eval_data_df,pd.get_dummies(eval_data_df['anatom_site_general_challenge'], drop_first=True, prefix='anatom')],axis=1)



eval_data_df.drop(['sex', 'anatom_site_general_challenge', 'patient_id'],axis=1, inplace=True)



# normalize the age...

column_names_to_normalize = ['age_approx']

x = eval_data_df[column_names_to_normalize].values

scaler = MinMaxScaler() 

x_scaled = scaler.fit_transform(x)

df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = eval_data_df.index)

eval_data_df[column_names_to_normalize] = df_temp



# take the vgg prediction and add it to the dataframe...

eval_data_df['vgg_prediction'] = eval_data_df['image_name'].apply(vgg_prediction_test)



# now we dont need the image name anymore...

eval_data_df = eval_data_df.drop(columns=['image_name'])

X = eval_data_df



xgb_X = xgb.DMatrix(X)

predictions_bin = model_bin.predict(xgb_X)



sub_data_bin = {'image_name': imageNames_lst, 'target':predictions_bin}

sub_df_bin = pd.DataFrame(sub_data_bin) 

sub_df_bin
sub_df_bin.to_csv('submission.csv', index=False)