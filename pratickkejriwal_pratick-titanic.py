!pip install tensorflow==2.0.0-alpha0 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import feature_column

from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
input_data = pd.read_csv("../input/train.csv")

X = input_data.drop(["PassengerId","Name","Ticket"],axis=1)
m = input_data.loc[(input_data['Sex'] >= "male")]

m_mean = m['Age'].mean()
f = input_data.loc[(input_data['Sex'] >= "female")]

f_mean = f['Age'].mean()
X['Age'] = np.where( (X['Age'].isnull()) & (X['Sex'] == 'male'), m_mean, X['Age'])

X['Age'] = np.where( (X['Age'].isnull()) & (X['Sex'] == 'female'), f_mean, X['Age'])



x_embarked_mode = X['Embarked'].mode()

X['Embarked'] = np.where(X['Embarked'].isnull(), x_embarked_mode , X['Embarked'])



X['Cabin'] = np.where( ~X['Cabin'].isnull() , X['Cabin'].str[0], 'M' )
train, val = train_test_split(X, test_size=0.2)
def df_to_dataset(df, shuffle=True, batch_size=32):

  df = df.copy()

  labels = df.pop('Survived')

  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

  if shuffle:

    ds = ds.shuffle(buffer_size=len(df))

  ds = ds.batch(batch_size)

  return ds
batch_size=16

train_ds = df_to_dataset(train, batch_size=batch_size)

val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
feature_columns = []



for ft in ['Age','SibSp','Parch','Fare']:

  feature_columns.append(feature_column.numeric_column(ft))



sex = feature_column.categorical_column_with_vocabulary_list('Sex', ['male', 'female'])

sex_one_hot = feature_column.indicator_column(sex)

feature_columns.append(sex_one_hot)



embarked = feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C','Q'])

emb_one_hot = feature_column.indicator_column(embarked)

feature_columns.append(emb_one_hot)



cabin = feature_column.categorical_column_with_vocabulary_list('Cabin',

        ['M', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']) 

cabin_one_hot = feature_column.indicator_column(cabin)

feature_columns.append(cabin_one_hot)



pclass = feature_column.categorical_column_with_identity('Pclass',num_buckets=4) 

pclass_one_hot = feature_column.indicator_column(pclass)

feature_columns.append(pclass_one_hot)

feature_layer = layers.DenseFeatures(feature_columns)
model = tf.keras.Sequential([feature_layer,

  layers.Dense(128, activation='relu'),

  layers.Dense(128, activation='relu'),

  layers.Dropout(rate=0.5),

  layers.Dense(128, activation='relu'),

  layers.Dense(128, activation='relu'),

  layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



model.fit(train_ds,validation_data=val_ds,epochs=35)

test_data = pd.read_csv("../input/test.csv")

test_X = test_data.drop(["PassengerId","Name","Ticket"],axis=1)
test_m = test_X.loc[(test_X['Sex'] >= "male")]

test_m_mean = test_m['Age'].mean()



test_f = test_X.loc[(test_X['Sex'] >= "female")]

test_f_mean = test_f['Age'].mean()



test_X['Age'] = np.where((test_X['Age'].isnull()) & (test_X['Sex'] == 'male'),test_m_mean,test_X['Age'])

test_X['Age'] = np.where((test_X['Age'].isnull()) & (test_X['Sex'] == 'female'),test_f_mean,test_X['Age'])



test_X['Cabin'] = np.where( ~test_X['Cabin'].isnull() , test_X['Cabin'].str[0], 'M' )
test_X = test_X.fillna(test_X.mean())
test_ds = tf.data.Dataset.from_tensor_slices(dict(test_X))

test_ds = test_ds.batch(16)
predictions = model.predict(test_ds)

predictions = np.where(predictions>0.5,1,0)

predictions = predictions.reshape(418)
sub = np.empty((418,2),dtype=int)

sub[:,0] = test_data["PassengerId"]

sub[:,1] = predictions
submit = pd.DataFrame(data=sub,columns=["PassengerId","Survived"])
submit.to_csv("submission.csv",index = False)