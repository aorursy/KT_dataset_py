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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from glob import glob
from tqdm import tqdm, tqdm_notebook
import random
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from numpy.random import seed
print(os.listdir('../input/skin-cancer-mnist-ham10000'))
path = '../input/skin-cancer-mnist-ham10000'
images_path = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join(path, '*', '*.jpg'))}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma',
    'sq': 'Squamous cell carcinoma'
}
skin_df = pd.read_csv(os.path.join(path,'HAM10000_metadata.csv'))
skin_df['path'] = skin_df['image_id'].map(images_path.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
skin_df.info()
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((71,71))))
skin_df.head()
# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()
X = skin_df['image']
y = skin_df['cell_type_idx']
X = X.values
X = X/255
X.shape
lst = []
for _ in X:
    lst.append(_)
X = np.array(lst)
print(X.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=28)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=0.20,random_state=28)
from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes = 7)
y_val = to_categorical(y_val , num_classes=7)
batch_size = 256
train_input_shape = (71, 71, 3)
n_classes = 7
from tensorflow.keras.layers import Input
# Load pre-trained model
#base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)
# input_tensor = Input(shape=(50,50,3))
base_model = Xception(include_top = False , input_shape = train_input_shape)


for layer in base_model.layers:
    layer.trainable = True
# Add layers at the end
model = base_model.output
model = Flatten()(model)

model = Dense(512, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Dense(128, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Dense(52, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

model = Dense(16, kernel_initializer='he_uniform')(model)
model = Dropout(0.2)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)

output = Dense(n_classes, activation='softmax')(model)

model = Model(inputs=base_model.input, outputs=output)
optimizer = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])
n_epoch = 10

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                              verbose=1, mode='auto')
history = model.fit(x_train,y_train,epochs=60,
                              callbacks=[reduce_lr,early_stop],validation_data=(x_val,y_val))
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
model.summary()
model2 = Model(model.input, model.layers[-7].output)
model2.summary()
predictions = model2.predict(X)
predictions[0]
predictions[10011]
len(predictions[0])
data_df = skin_df
complete_data = pd.concat([data_df, pd.DataFrame(predictions)], axis=1)
complete_data.head()
### saving a model!!!!!

model_json = model2.to_json()
with open("model_v2.json", "w") as json_file:
    json_file.write(model_json)
### saving a model!!!!!

completemodel_json = model.to_json()
with open("completemodel1.json", "w") as json_file:
    json_file.write(completemodel_json)
model2.save_weights("model_v2_weights.h5")
print("Saved model to disk")
model.save_weights("completeweights.h5")
print("Saved model to disk")
complete_data.head()
complete_data.columns
dxtype_df=pd.get_dummies(complete_data['dx_type'],drop_first=False)
complete_data=pd.concat([dxtype_df,complete_data],axis=1)
# complete_data.drop(['dx_type'],axis=1,inplace=True)
complete_data.head()
localization_df=pd.get_dummies(complete_data['localization'],drop_first=False)
complete_data=pd.concat([localization_df,complete_data],axis=1)
# complete_data.drop(['dx_type'],axis=1,inplace=True)
complete_data.head()
sex_df=pd.get_dummies(complete_data['sex'],drop_first=False)
sex_df.drop(['unknown'],axis=1,inplace=True)
complete_data=pd.concat([sex_df,complete_data],axis=1)
# complete_data.drop(['dx_type'],axis=1,inplace=True)
complete_data.head()
complete_data.columns
X_labels = complete_data.drop(['lesion_id','image_id','dx_type','dx','path','cell_type','cell_type_idx','sex','path','localization','image'],axis=1,inplace=False)
y_label = complete_data['cell_type_idx']
X_labels.head()
complete_data.to_csv('skin_data_v2.csv')
preds = model.predict(x_test)
preds = model.predict(x_test)
lst = []
for a in preds:
    lst.append(np.argmax(a))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
confusion_matrix(lst,y_test)
accuracy_score(lst,y_test)
print(classification_report(y_test,lst))
skin_df = pd.read_csv('skin_data_v2.csv')
skin_df.drop(['Unnamed: 0'],axis=1,inplace=True)
skin_df.head()
!pip install pycaret
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
skin_df.head()
X_labels = skin_df.drop(['lesion_id','image_id','dx','dx_type','sex','localization','path','cell_type','cell_type_idx','image'],axis=1,inplace=False)
y_label = skin_df['cell_type_idx']
data_classification = pd.concat([X_labels,y_label],axis=1)
data_classification.fillna(data_classification['age'].mean(),inplace=True)
# import the classification module 
from pycaret import classification
# setup the environment 
classification_setup = classification.setup(data= data_classification, target='cell_type_idx')
# build the decision tree model
classification_dt = classification.create_model('dt')
# build the xgboost model
classification_xgb = classification.create_model('xgboost')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_labels,y_label,train_size=0.8,random_state=40)
import xgboost as xgb
clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree',
                                            colsample_bylevel=1,
                                            colsample_bynode=1,
                                            colsample_bytree=1, gamma=0,
                                            learning_rate=0.1, max_delta_step=0,
                                            max_depth=3, min_child_weight=1,
                                            missing=None, n_estimators=100,
                                            n_jobs=-1, nthread=None,
                                            objective='binary:logistic',
                                            random_state=1855, reg_alpha=0,
                                            reg_lambda=1, scale_pos_weight=1,
                                            seed=None, silent=None, subsample=1,
                                            verbosity=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))