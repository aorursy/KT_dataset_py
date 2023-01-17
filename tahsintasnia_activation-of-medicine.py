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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Import data 
import pandas as pd

train_features = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")
train_features.drop(['sig_id'], axis='columns', inplace=True)

train_target_full = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
train_target = train_target_full.copy()
train_target.drop(['sig_id'], axis='columns', inplace=True)

test_features_full = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")
test_features = test_features_full.copy()
test_features.drop(['sig_id'], axis='columns', inplace=True)
#Label encoding 
from sklearn.preprocessing import LabelEncoder

label_train_features = train_features.copy()
label_test_features = test_features.copy()

object_cols = ['cp_type', 'cp_time', 'cp_dose']
label_encoder = LabelEncoder()
for col in object_cols:
    label_train_features[col] = label_encoder.fit_transform(train_features[col])
    label_test_features[col] = label_encoder.transform(test_features[col])

# Check the encoded features
label_test_features.loc[0:5,['cp_type', 'cp_time', 'cp_dose']]
test_features.loc[0:5,['cp_type', 'cp_time', 'cp_dose']]
#model creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dropout(0.2, input_shape=(n_inputs,)))
    model.add(Dense(306, input_dim=875, kernel_initializer='he_uniform', activation='sigmoid'))
    model.add(Dropout(0.5)) 
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(856, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(206, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics = ['accuracy'])
    return model

n_inputs = label_train_features.shape[1]
n_outputs=train_target.shape[1]
model = get_model(n_inputs, n_outputs)
#deployment
#model.fit(label_train_features, train_target, verbose=1, epochs=100, validation_split=0.2, batch_size=32)


#predictions = model.predict(label_test_features)
#submission
data = pd.DataFrame.from_records(predictions)
data.insert(0,'sig_id',test_features_full['sig_id'])
data.columns = train_target_full.columns
data.to_csv('submission.csv', index=False)