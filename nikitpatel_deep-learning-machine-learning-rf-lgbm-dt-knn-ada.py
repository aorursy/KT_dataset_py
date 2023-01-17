# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing

from sklearn.model_selection import StratifiedShuffleSplit,train_test_split



import keras

import tensorflow as tf

from keras.models import Sequential,load_model,Model

from keras.optimizers import *

from keras.utils import to_categorical

from keras.layers import *

from keras.callbacks import *

from keras import backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers, layers

sess = tf.Session()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

K.tensorflow_backend._get_available_gpus()

K.set_session(sess)



# Any results you write to the current directory are saved as output.
xtrain = pd.read_csv('../input/X_train.csv')

ytrain = pd.read_csv('../input/y_train.csv')

test=pd.read_csv("../input/X_test.csv")
### feature extraction of orientation, angular_velocity, linear_acceleration, velocity_to_acceleration and velocity_linear_acceleration

def feature_extraction(raw_frame):

    raw_frame['orientation'] = raw_frame['orientation_X'] + raw_frame['orientation_Y'] + raw_frame['orientation_Z']+ raw_frame['orientation_W']

    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']

    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

    raw_frame['velocity_linear_acceleration'] = raw_frame['linear_acceleration'] * raw_frame['angular_velocity']

    return raw_frame
xtrain = feature_extraction(xtrain)

test = feature_extraction(test)
### more feature extraction with mean, mode, std, variance, min, max and so on...



def feature_extraction_more(raw_frame):

    frame = pd.DataFrame([])

    for col in raw_frame.columns[3:]:

        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()

        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()

        frame[col + '_var'] = raw_frame.groupby(['series_id'])[col].var()

        frame[col + '_sem'] = raw_frame.groupby(['series_id'])[col].sem()

        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()

        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()

        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']

        frame[col + '_max_minus_min'] = frame[col + '_max'] - frame[col + '_min']

        frame[col + '_std_to_var'] = frame[col + '_std'] * frame[col + '_var']

        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return frame
train_df = feature_extraction_more(xtrain)

test_df = feature_extraction_more(test)
print("train shape",train_df.shape)

print("test shape", test_df.shape)
scaler = preprocessing.StandardScaler()

# Apply transform to both the training set and the test set.

train_df = scaler.fit_transform(train_df)

test_df = scaler.fit_transform(test_df)
### lable encoding 

le = preprocessing.LabelEncoder()

le.fit(ytrain.surface)

ytrain['surface'] = le.transform(ytrain.surface)

train_label = to_categorical(ytrain['surface'])

train_label.shape
train_x,val_x,train_y,val_y = train_test_split(train_df, train_label, test_size = 0.10, random_state=14)

train_x.shape,val_x.shape,train_y.shape,val_y.shape
nb_features = train_df.shape[1]

nb_out = train_label.shape[1]

nb_features,nb_out
## https://www.kaggle.com/kabure/titanic-eda-keras-nn-pipelines

## Creating the model

model = Sequential()



# Inputing the first layer with input dimensions

model.add(Dense(165, 

                activation='relu',  

                input_dim = nb_features,

                kernel_initializer='uniform'))



# Adding an Dropout layer to previne from overfitting

model.add(Dropout(0.50))



#adding second hidden layer 

model.add(Dense(60,

                kernel_initializer='uniform',

                activation='relu'))



# Adding another Dropout layer

model.add(Dropout(0.50))



# adding the output layer that is binary [0,1]

model.add(Dense(nb_out, activation='softmax'))



#Visualizing the model

model.summary()
model_feat = Model(inputs=model.input,outputs=model.get_layer('dropout_2').output)
train_feature = model_feat.predict(train_x)

val_feature = model_feat.predict(val_x)

test_feature = model_feat.predict(test_df)
feat_train.shape
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import mean_absolute_error

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



para={'boosting_type': 'gbdt',

 'colsample_bytree': 0.85,

 'learning_rate': 0.1,

 'max_bin': 512,

 'max_depth': -1,

 'metric': 'multi_error',

 'min_child_samples': 8,

 'min_child_weight': 1,

 'min_split_gain': 0.5,

 'nthread': 3,

 'num_class': 9,

 'num_leaves': 31,

 'objective': 'multiclass',

 'reg_alpha': 0.8,

 'reg_lambda': 1.2,

 'scale_pos_weight': 1,

 'subsample': 0.7,

 'subsample_for_bin': 200,

 'subsample_freq': 1}



Classifier = [

    

        LGBMClassifier(),

        LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

        KNeighborsClassifier(),

        SVC(kernel="rbf", C=0.025, probability=True),

        DecisionTreeClassifier(),

        RandomForestClassifier(n_estimators=500,max_depth=20, min_samples_split=5,

                             class_weight='balanced'),

        AdaBoostClassifier(),

        GaussianNB(),

]
Accuracy=[]

Model=[]



for classifier in Classifier:

    try:

        

        fit = classifier.fit(train_feature,np.argmax(train_y,axis=1))

        pred = fit.predict(val_feature)

    except Exception:

        fit = classifier.fit(train_feature,np.argmax(train_y,axis=1))

        pred = fit.predict(val_feature)

        

        

    score = accuracy_score(np.argmax(val_y,axis=1), pred)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+' is '+str(score))

    
classifier = KNeighborsClassifier()

fit = classifier.fit(train_feature,np.argmax(train_y,axis=1))
pred = fit.predict(test_feature)
submission = pd.read_csv("../input/sample_submission.csv")

submission['surface'] = le.inverse_transform(pred)

submission.to_csv('knn.csv', index=False)