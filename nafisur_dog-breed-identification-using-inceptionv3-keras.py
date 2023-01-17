from keras.layers import Dense,Dropout,Input,GlobalAveragePooling2D,MaxPooling2D
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3 as pickleModel,preprocess_input
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam,SGD
from keras.preprocessing.image import img_to_array,load_img,ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_colwidth=150

df1=pd.read_csv('../input/labels.csv')
df1.head()
img_file='../input/train/'
df2=df1.assign(img_path=lambda x: img_file + x['id'] +'.jpg')
df2.head()
df2.breed.value_counts()
#Top 20 breed
top_20=list(df2.breed.value_counts()[0:20].index)
top_20
df3=df2[df2.breed.isin(top_20)]
df3.shape
df=df3.copy()
df.shape
%%time
img_pixel = np.array([np.array(Image.open(filename).resize((299,299))) for filename in df['img_path']])
print(img_pixel.shape)
img_label=df.breed
img_label=pd.get_dummies(df.breed)
img_label.head()
X=img_pixel
y=img_label.values
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
inputshape=X_train[0].shape
num_class=len(y[0])
print(inputshape)
print(num_class)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
num_train_sample=len(X_train)
print(num_train_sample)
num_test_sample=len(X_test)
print(num_test_sample)
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
                                 rescale=1./255,
                                 rotation_range=30, width_shift_range=0.2, 
                                 height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,horizontal_flip=True
                                )
test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input,
                               rescale=1./255)
%%time
training_set=train_datagen.flow(X_train,y=y_train,batch_size=32)

%%time
testing_set=test_datagen.flow(X_test,y=y_test,batch_size=32)

# base_model = pickleModel(weights = 'imagenet', include_top = False, input_shape=inputshape)
# # Add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # Add a fully-connected layer and a logistic layer with 20 classes 
# #(there will be 120 classes for the final submission)
# x = Dense(512, activation='relu')(x)
# predictions = Dense(num_class, activation='softmax')(x)
# # The model we will train
# model = Model(inputs = base_model.input, outputs = predictions)
# # first: train only the top layers i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False
# # Compile with Adam
# model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# history=model.fit_generator(training_set,
#                       steps_per_epoch = num_train_sample//32,
#                       validation_data = testing_set,
#                       validation_steps = num_test_sample//32,
#                       epochs = 1,
#                       verbose = 1)
