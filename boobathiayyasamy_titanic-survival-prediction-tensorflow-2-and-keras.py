# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



#Installing tensorflow 2.0

#!pip install -q tensorflow==2.0.0-alpha0



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanic_train = pd.read_csv('/kaggle/input/train.csv')

titanic_test = pd.read_csv('/kaggle/input/test.csv')



titanic_train.head()
titanic_test.head()
titanic_train.info()
titanic_train.describe()
from sklearn.model_selection import train_test_split



X_train, X_val = train_test_split(titanic_train, test_size=0.2, random_state=42)
X_train.head()
X_train['Embarked'].value_counts()
embarked_replace = 'S'

Y_train = X_train['Survived']

Y_val = X_val['Survived']



dropping_columns = ['Survived', 'Name', 'Ticket', 'Cabin']

X_train = X_train.drop(dropping_columns, axis=1)

X_val = X_val.drop(dropping_columns, axis=1)
X_train['Embarked'] = X_train['Embarked'].fillna(embarked_replace)

X_val['Embarked'] = X_val['Embarked'].fillna(embarked_replace)
X_train.head()
X_val.head()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([        

                          ('imputer', SimpleImputer(strategy="mean")),        

                          ('std_scaler', StandardScaler()),

                          

                        ])



fare_pipeline = Pipeline([        

                          ('rb_scaler', RobustScaler()),    

                        ])
num_attribs = ['Pclass','Age','SibSp','Parch']

fare_attrib = ['Fare']

cat_attribs = ['Sex','Embarked']
full_pipeline = ColumnTransformer([        

    ("num", num_pipeline, num_attribs),        

    ("f_num", fare_pipeline, fare_attrib),   

    ("cat", OneHotEncoder(), cat_attribs)   

])
X_train_prepared = full_pipeline.fit_transform(X_train)
X_val_prepared = full_pipeline.transform(X_val)
X_train_prepared.shape
X_train_prepared
Y_val.head()
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint

#from keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.callbacks import EarlyStopping
tf.__version__
keras.backend.clear_session()

np.random.seed(42)

#tf.random.set_seed(42)



model = keras.models.Sequential([       

                                    keras.layers.Dense(90, activation="relu", input_dim=10),    

                                    keras.layers.Dropout(0.2),                                     

                                    keras.layers.Dense(45, activation="relu"),    

                                    keras.layers.Dropout(0.2),                                     

                                    keras.layers.Dense(15, activation="relu"),    

                                    keras.layers.Dense(1, activation="sigmoid"),

                                ])



model.compile(loss="binary_crossentropy",              

              optimizer="sgd",

              #optimizer="RMSprop",

              metrics=["accuracy"])

model.summary()
filepath = "titanic_model.h5"



#lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')



eraly_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')

# Reducing the Learning Rate if result is not improving. 

reduce_lr = ReduceLROnPlateau(monitor='val_loss', min_delta=0.0004, patience=2, factor=0.1, min_lr=1e-6, mode='auto',

                              verbose=1)
history = model.fit(X_train_prepared, Y_train, epochs=500,callbacks = [eraly_stop, reduce_lr, checkpoint],

                    validation_data=(X_val_prepared, Y_val))
history.params
history.history.keys()
pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

#save_fig("keras_learning_curves_plot")

plt.show()
test_dropping_columns = ['Name', 'Ticket', 'Cabin']



X_test =  titanic_test.drop(test_dropping_columns, axis=1)

X_test['Embarked'] = X_test['Embarked'].fillna(embarked_replace)
X_test_prepared = full_pipeline.transform(X_test)
titanic_model = keras.models.load_model("titanic_model.h5")

Y_test_predict_DL = titanic_model.predict_classes(X_test_prepared)
Y_test_predict_DL
#Writing to File

submission=pd.DataFrame(titanic_test.loc[:,['PassengerId']])

submission['Survived']=Y_test_predict_DL

#Any files you save will be available in the output tab below



submission.to_csv('submission.csv', index=False)