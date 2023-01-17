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
import pandas as pd
from sklearn.model_selection import train_test_split
all_df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
all_df.head()
feature_column_names = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction'
,'high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']
label_column_name = ['DEATH_EVENT']
y = all_df.pop('DEATH_EVENT')
train, test, train_y, test_y = train_test_split(all_df, y, test_size=0.25, random_state=2, shuffle=True)
feature_columns = []
for feature in feature_column_names:
  feature_columns.append(tf.feature_column.numeric_column(feature,dtype=tf.float32))
def make_input_fn(features,label,shuffle=False,batch_size=32):
  def input_func():
    ds = tf.data.Dataset.from_tensor_slices((dict(features),label))
    if shuffle:
      ds = ds.shuffle(500).repeat()
    ds = ds.batch(batch_size)
    return ds
  return input_func
train_input_func = make_input_fn(train,train_y,shuffle=True)
test_input_func = make_input_fn(test,test_y,shuffle=False)
classifier = tf.estimator.DNNClassifier(hidden_units=[30,10],feature_columns=feature_columns,n_classes=2,
                                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
classifier.train(train_input_func,steps=10000)
eval_result = classifier.evaluate(test_input_func)
print(eval_result['accuracy'])
def input_func_pred(features,batch_size=32):
  return tf.data.Dataset.from_tensor_slices((dict(features))).batch(batch_size)

predict = {}
print("set your values with this order, all values should be float, don't forget the '.' ")
for f in feature_column_names:
    print(f)
    
for feature in feature_column_names:
  valid = True
  while valid:
    val = input(feature+" = ")
    if not val.isdigit():
      valid = False
  predict[feature] = [float(val)]

pred = classifier.predict(input_fn=lambda : input_func_pred(predict))
for p in pred:
  class_id = p['class_ids'][0]
  probability_death = p['probabilities'][1]
  probability_live = p['probabilities'][0]
  print(p)
  print("death probability : ")
  print(probability_death*100)
  print("live probability : ")
  print(probability_live*100)