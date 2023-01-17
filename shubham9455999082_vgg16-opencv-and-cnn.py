# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import openslide
import os
import cv2
import PIL
from IPython.display import Image, display
from keras.applications.vgg16 import VGG16,preprocess_input
import plotly.graph_objs as go
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.layers import GlobalMaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
import skimage.io
from sklearn.model_selection import KFold
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import tensorflow as tf
from tensorflow.python.keras import backend as K
sess = K.get_session()
train=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
sub=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
sub.head()
labels=[]
data=[]
data_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train'
for i in range(train.shape[0]):
    data.append(data_dir + train['image_name'].iloc[i]+'.jpg')
    labels.append(train['target'].iloc[i])
df=pd.DataFrame(data)
df.columns=['images']
df['target']=labels
df
from keras.preprocessing import image
'''
train_image = []
for i in range(train.shape[0]):
    img = image.load_img('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name'].iloc[i]+'.jpg', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
'''
'''
import glob
import cv2
import numpy as np
pic_num=1
IMG_DIR='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'
def read_images(directory):
    for img in glob.glob(directory+"/*.jpg"):
        image = cv2.imread(img)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        train_image = cv2.resize(image/255.0  , (32 , 32))
        #cv2.imwrite("small/"+str(pic_num)+'.jpg',resized_img)

        yield train_image

train_image =  np.array(list(read_images(IMG_DIR)))
'''
train
train
df
train.head()
train['target'].value_counts()
test.head()
df
X_train, X_val, y_train, y_val = train_test_split(df['images'],df['target'], test_size=0.2, random_state=1234)
y_train
train=pd.DataFrame(X_train)
train.columns=['images']
train['target']=y_train

validation=pd.DataFrame(X_val)
validation.columns=['images']
validation['target']=y_val

train['target']=train['target'].astype(str)
validation['target']=validation['target'].astype(str)
train
validation
train_datagen = ImageDataGenerator()
                                   #,rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,horizontal_flip=True)
val_datagen=train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_dataframe(
    train,
    x_col='images',
    y_col='target',
    target_size=(224, 224)
    #batch_size=32,
    #class_mode='categorical'
)

validation_generator = val_datagen.flow_from_dataframe(
    validation,
    x_col='images',
    y_col='target',
    target_size=(224, 244)
    #batch_size=32,
    #class_mode='categorical'
)
from keras.applications import VGG16

# include top should be False to remove the softmax layer
pretrained_model = VGG16(include_top=False, weights='imagenet')
pretrained_model.summary()
def vgg16_model( num_classes=None):

    model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x=Flatten()(model.output)
    x=Dropout(0.5)(x)
    output=Dense(num_classes,activation='softmax')(x)
    model=Model(model.input,output)
    return model

vgg_conv=vgg16_model(6)
vgg_conv.summary()
def kappa_score(y_true, y_pred):
    
    y_true=tf.math.argmax(y_true)
    y_pred=tf.math.argmax(y_pred)
    return tf.compat.v1.py_func(cohen_kappa_score ,(y_true, y_pred),tf.double)
opt = SGD(0.001,momentum=0.9,decay=1e-4)
vgg_conv.compile(loss='categorical_crossentropy',optimizer=opt,metrics=[kappa_score])
nb_epochs = 5
batch_size=32
nb_train_steps = train.shape[0]//batch_size
nb_val_steps=validation.shape[0]//batch_size
print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
vgg_conv.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_steps,
    epochs=nb_epochs,
    validation_data=validation_generator,
    validation_steps=nb_val_steps)
# submission code from https://www.kaggle.com/frlemarchand/high-res-samples-into-multi-input-cnn-keras
def predict_submission(df, path):
    
    df["image_path"] = [path+image_id+".tiff" for image_id in df["image_name"]]
    df["target"] = 0
    predictions = []
    for idx, row in df.iterrows():
        print(row.image_path)
        img=skimage.io.imread(str(row.image_path))
        img = cv2.resize(img, (224,224))
        img = cv2.resize(img, (224,224))
        img = img.astype(np.float32)/255.
        img=np.reshape(img,(1,224,224,3))
        prediction=vgg_conv.predict(img)
        predictions.append(np.argmax(prediction))
            
    df["target"] = predictions
    df = df.drop('image_path', 1)
    return df[["image_name","target"]]
test_path = "'/kaggle/input/siim-isic-melanoma-classification/jpeg/train"
submission_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv")

if os.path.exists(test_path):
    test_df = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/test.csv")
    submission_df = predict_submission(test_df, test_path)

submission_df.to_csv('submission.csv', index=False)
submission_df.head()
from keras.utils import to_categorical
# extract train and val features
vgg_features_train = pretrained_model.predict(train_generator)
vgg_features_val = pretrained_model.predict(validation_generator)
vgg_features_train
model2 = Sequential()
model2.add(Flatten(input_shape=(7,7,512)))
model2.add(Dense(100, activation='relu'))
model2.add(Dropout(0.5))
model2.add(BatchNormalization())
model2.add(Dense(10, activation='softmax'))

# compile the model
model2.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

model2.summary()

# train model using features generated from VGG16 model

#model2.fit(vgg_features_train, epochs=50,  validation_data=vgg_features_val)



