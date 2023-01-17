import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import cv2

#Loading training dataset 

#image_path = '../input/siim-isic-melanoma-classification/jpeg/'

submission=pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

dataset_path = '/kaggle/input/siim-isic-melanoma-classification/'



train_df = pd.read_csv(dataset_path+'train.csv')

train_df.head()
#Loading test dataset 

test_df = pd.read_csv(dataset_path+'test.csv')

test_df.head()
label = []

data = []

for i in range(train_df.shape[0]):

    data.append(dataset_path+'jpeg/train/'+train_df['image_name'].iloc[i]+'.jpg')

    label.append(train_df['target'].iloc[i])

train_image = pd.DataFrame(data)

train_image.columns = ['images']

train_image['target'] = label



print (train_image['images'])

test_data = []

for i in range(test_df.shape[0]):

    test_data.append(dataset_path+'jpeg/test/'+test_df['image_name'].iloc[i]+'.jpg')

test_image = pd.DataFrame(test_data)

test_image.columns = ['images']



print (test_image['images'])
X_train, X_val, y_train, y_val = train_test_split(train_image['images'],train_image['target'], test_size=0.2, random_state=1234)



train=pd.DataFrame(X_train)

train.columns=['images']

train['target']=y_train



validation=pd.DataFrame(X_val)

validation.columns=['images']

validation['target']=y_val
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.layers import Flatten, Dense

from tensorflow.keras import Model
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,horizontal_flip=True)

val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(

    train_image,

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
nb_epochs = 2

batch_size=64

nb_train_steps = train_image.shape[0]//batch_size

nb_val_steps=validation.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps,nb_val_steps))
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x=Flatten()(model.output)

output=Dense(1,activation='sigmoid')(x) # because we have to predict the AUC

model=Model(model.input,output)
model.compile(loss='binary_crossentropy', metrics=['accuracy'],optimizer='Adam')


model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_steps,

    epochs=nb_epochs,

    validation_data=validation_generator,

    validation_steps=nb_val_steps)
target=[]

for path in test_image['images']:

    img=cv2.imread(str(path))

    

    img = cv2.resize(img, (224,224))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)/255.

    img=np.reshape(img,(1,224,224,3))

    prediction=model.predict(img)

    target.append(prediction[0][0])



submission['target']=target
submission.to_csv('submission.csv', index=False)

submission.head()