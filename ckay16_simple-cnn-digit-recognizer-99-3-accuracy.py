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
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.svm import SVC
data = pd.read_csv("../input/digit-recognizer/train.csv")  #read the training data
data.head()       #see the data
data.info()           #see how many entries and type of data
data_y = data['label']                               # get dependent variable 
data.drop(['label'],axis=1,inplace=True)             # drop it from original dataframe
data_x = data                                        # store all independent variables separately

data_y.head()
data_x
data_x = data_x.values.reshape(-1,28,28,1)                     # reshape to proper shape
data_y = data_y.values

from keras.utils.np_utils import to_categorical               # turn individual numbers into categorical data
data_y = to_categorical(data_y)                               # Ex : [2] -> [0,0,1,0,0,0,0,0,0,0]
data_x = data_x / 255.0                        # since values are between 0 and 255, divide by 255 to make them between 0 to 1, easier for processing
model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),       # adding convolution layer with input size (28,28,1) , 1 means the images are in greyscale not rgb
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),                                              # adding pooling layer
        tf.keras.layers.Dropout(0.5),
    
        tf.keras.layers.Conv2D(128,(3,3),activation='relu'),  
        tf.keras.layers.Conv2D(192,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.5),
    
    
        tf.keras.layers.Flatten(),                                                      # flatten will flatten the input (28,28,1) to a single array
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(512,activation='relu',kernel_regularizer=l2(0.01)),                                   # hidden layer with 256 units
        #tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10,activation='softmax')                                  # output layer with 10 units, each representing the corresponding output
])
model.summary()
model.compile(Adam(lr=0.001),loss="categorical_crossentropy",metrics=['accuracy'])          
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, 
                                            factor=0.5, min_lr=0.00001)
from tensorflow.keras.preprocessing.image import ImageDataGenerator                #for Data Augmentation
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x,data_y,test_size=0.2,random_state=0)         # splitting data into train and test for validation later on
datagen = ImageDataGenerator(rotation_range=10,                                    # rotate image by 10 degrees
      width_shift_range=0.1,                                                       # shift width focus randomly by 0.1
      height_shift_range=0.1,                                                      #shift height focus randomly by 0.1
      shear_range=0.1,                                                             # shear the image by a factor of 0.15
      zoom_range=0.2,                                                              # zoom into the image by a factor of 0.25
      fill_mode='nearest')                                                         # if pixels are missings fill them by taking their nearest pixels

train_datagen = datagen.flow(train_x,train_y,batch_size=50)                        # flow training data in batches of 50 to the model

validation_datagen = datagen.flow(test_x,test_y,batch_size=50)                     # flow validation data in batches of 50
model.fit(train_datagen,
          validation_data = validation_datagen,
          steps_per_epoch=500,
          epochs=50
         )                                                                       # fit the model with training and validation generators

model.save('digitrec.h5')
cur_model = tf.keras.models.load_model('/kaggle/input/digitrec/ogdigitrec1.h5')
cur_model.summary()
new_model = tf.keras.Model(inputs=cur_model.input,outputs=cur_model.get_layer('dense_1').output)
new_model.summary()

test = pd.read_csv("../input/digit-recognizer/test.csv")
test = test.values.reshape(-1,28,28,1)
test = test/255.0
preds = cur_model.predict(test)                    # use model to predict for all test values
preds
pred_labels = cur_model.predict_classes(test)
pred_labels
submission = np.argmax(preds,axis=1)                               # since results are in categorical form, choose the highest in each and store them 
submission                                                         # the above cell also gives the same output, you can use either one
my_submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':submission })               # make dataframe with the column headders and predicted values

my_submission.to_csv("results.csv",index=False)
my_submission.head()
import pandas as pd
submission = pd.read_csv('../input/digitrec/newresults1.csv')
submission.to_csv('newresults.csv',index=False)
svm_train = new_model.predict(train_x)
svm_val = new_model.predict(test_x)
svm_test = new_model.predict(test)
svm = SVC(kernel='rbf')

svm.fit(svm_train,np.argmax(train_y,axis=1))
svm.score(svm_train,np.argmax(train_y,axis=1))
svm.score(svm_val,np.argmax(test_y,axis=1))
svm_pred = svm.predict(svm_test)
my_submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':svm_pred })               # make dataframe with the column headders and predicted values

my_submission.to_csv("results.csv",index=False)
import xgboost as xgb

xb = xgb.XGBClassifier()

xb.fit(svm_train,np.argmax(train_y,axis=1))
xb.score(svm_train,np.argmax(train_y,axis=1))
xb.score(svm_val,np.argmax(test_y,axis=1))
xb_pred = xb.predict(svm_test)
xb_pred
my_submission = pd.DataFrame({'ImageId': range(1,len(test)+1) ,'Label':xb_pred })               # make dataframe with the column headders and predicted values

my_submission.to_csv("fin_results.csv",index=False)
