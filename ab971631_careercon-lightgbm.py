# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/X_train.csv")

test=pd.read_csv("../input/X_test.csv")

label=pd.read_csv("../input/y_train.csv")
test.head()
def feature_extraction(raw_frame):

    frame = pd.DataFrame()

    raw_frame['orientation'] = raw_frame['orientation_X'] + raw_frame['orientation_Y'] + raw_frame['orientation_Z']+ raw_frame['orientation_W']

    

    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Y']

    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

    raw_frame['velocity_linear_acceleration'] = raw_frame['linear_acceleration'] * raw_frame['angular_velocity']

    for col in raw_frame.columns[3:]:

        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()

        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()

        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()

        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()

        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']

        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return frame
train = feature_extraction(train)

test = feature_extraction(test)

train.head()
test.shape,train.shape
label=label["surface"]
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

label_encoder.fit(label)

label=label_encoder.transform(label)
Y_train = label

X_train = train

test=test

X_train.shape, Y_train.shape

scaler = StandardScaler()

scaler.fit(X_train)

# Apply transform to both the training set and the test set.

train_img = scaler.transform(X_train)

test_img = scaler.transform(test)


from sklearn.model_selection import GridSearchCV,StratifiedKFold, train_test_split

kfold = StratifiedKFold(n_splits=5)

from sklearn.metrics import roc_auc_score,roc_curve,auc

train_x,val_x,train_y,val_y = train_test_split(X_train, Y_train, test_size = 0.10, random_state=14)

train_x.shape,val_x.shape,train_y.shape,val_y.shape
import lightgbm

train_data = lightgbm.Dataset(train_x, label=train_y)

test_data = lightgbm.Dataset(val_x, label=val_y)
#used tuned parameters after applying gridsearch

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



model = lightgbm.train(para,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=50,

                      )
y_pred = model.predict(test)
class_prediction=pd.DataFrame(y_pred).idxmax(axis=1) 
submission1 = pd.DataFrame({

        "series_id": test.index,

        "surface": class_prediction,

        

    })

submission1.surface.value_counts()
submission1.surface=label_encoder.inverse_transform(submission1.surface)
submission1.surface.value_counts()
submission1.to_csv('submission1.csv', index=False)