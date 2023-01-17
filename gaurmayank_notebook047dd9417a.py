import pandas as pd
import numpy as np
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
test_features = pd.read_csv('../input/lish-moa/test_features.csv')

ss = pd.read_csv('../input/lish-moa/sample_submission.csv')
ss_lr = ss.copy()

cols = [c for c in ss.columns.values if c != 'sig_id']
train_targets['binary'] = train_targets[train_targets.columns[1:]].apply(
    lambda x: ''.join(x.astype(str)),
    axis=1
)

output_classes = len(train_targets['binary'].unique())  
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
le4 = LabelEncoder()

le1.fit(train_features['cp_type'])
le2.fit(train_features['cp_dose'])
le3.fit(train_features['cp_time'])


le4.fit(train_targets['binary'])


train_features['cp_type'] = le1.transform(train_features['cp_type'])
train_features['cp_dose'] = le2.transform(train_features['cp_dose'])
train_features['cp_time'] = le3.transform(train_features['cp_time'])


train_targets['binary_le'] = le4.transform(train_targets['binary'])

import tensorflow as tf

X = np.asarray(train_features.iloc[:,1:].values).astype(np.float32)
Y = train_targets['binary_le'].values
Y = tf.keras.utils.to_categorical(Y, output_classes)
#X,Y =np.asarray(train_features[:])[:,1:].astype(np.float32),np.asarray(train_targets['binary']).astype(np.float32)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(875, activation = tf.nn.relu),
    tf.keras.layers.Dense(400, activation=tf.nn.elu),
    tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(200, activation=tf.nn.relu),
    tf.keras.layers.Dense(100, activation=tf.nn.elu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(200, activation=tf.nn.relu),
    tf.keras.layers.Dense(328, activation=tf.nn.softmax),
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(X,Y, batch_size= 256, epochs =100)



test_features['cp_type'] = le1.transform(test_features['cp_type'])
test_features['cp_dose'] = le2.transform(test_features['cp_dose'])
test_features['cp_time'] = le3.transform(test_features['cp_time'])


X_test = np.asarray(test_features.iloc[:,1:].values).astype(np.float32)
y_pred = np.argmax(model.predict(X_test) , axis = 1)
y_pred_bin = list(le4.inverse_transform(y_pred))
y_submission = pd.DataFrame()
y_submission['sig_id'] = test_features['sig_id']
y_submission['pred'] = pd.Series(y_pred_bin)
y_submission['pred'] = y_submission['pred'].apply(list)
y_submission[cols] = pd.DataFrame(y_submission['pred'].tolist(), index= y_submission.index)
del y_submission['pred']
y_submission.to_csv('submission.csv', index = False)