# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd
train_data_df = pd.read_csv('../input/titanic/train.csv')

test_data_df = pd.read_csv('../input/titanic/test.csv')
print(train_data_df.shape)

print(test_data_df.shape)

print(train_data_df.columns)

print(train_data_df.isna().sum())

print(test_data_df.isna().sum())

feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



train_features = train_data_df[feature_names]

test_features = test_data_df[feature_names]

train_features.head()

test_features.head()
def fill_missing_data(df, category):

    df[category].fillna(df[category].dropna().median(), inplace=True)

    

fill_missing_data(train_features, 'Age')

fill_missing_data(test_features, 'Age')



train_features['Embarked'].fillna('S', inplace=True)

test_features['Embarked'].fillna('S', inplace=True)



fill_missing_data(train_features, 'Fare')

fill_missing_data(test_features, 'Fare')

def encode_sex(df):

    return pd.concat([df, pd.get_dummies(df["Sex"], prefix="Sex")], axis=1).drop(["Sex"], axis=1)

    

train_features = encode_sex(train_features)

test_features = encode_sex(test_features)

    

train_features.head()

test_features.head()
def encode_embarked(df):

    return pd.concat([df, pd.get_dummies(df["Embarked"], prefix="Embarked")], axis=1).drop(["Embarked"], axis=1)

    

train_features = encode_embarked(train_features)

test_features = encode_embarked(test_features)

    

train_features.head()

test_features.head()
train_targets = train_data_df['Survived']

train_targets.head()
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=1)

# model.fit(train_features, train_target)
# test_predictions = model.predict(test_features)
# output = pd.DataFrame({'PassengerId': test_data_df['PassengerId'], 'Survived': test_predictions})

# output['Survived'].loc[output['Survived'] < 0.5] = 0

# output['Survived'].loc[output['Survived'] > 0.5] = 1

# output['Survived'] = output['Survived'].astype('int32')

# print(output)
# output.to_csv('submission.csv', index=False)
import tensorflow as tf

print(tf.__version__)
print(train_features.dtypes)

print(test_features.dtypes)
def get_model():

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(512, activation='relu'))

    model.add(tf.keras.layers.Dense(256, activation='relu'))

    model.add(tf.keras.layers.Dense(128, activation='relu'))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adamax', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    return model



model = get_model()

history = model.fit(train_features, train_targets, epochs=25, shuffle=True, validation_split=0.3)
print('loss =', history.history['loss'][-1])

print('accuracy =', history.history['accuracy'][-1])



print('val_loss =', history.history['val_loss'][-1])

print('val_accuracy =', history.history['val_accuracy'][-1])
test_predictions = model.predict(test_features)
output = pd.DataFrame({'PassengerId': test_data_df['PassengerId'], 'Survived': [p[0] for p in test_predictions]})

output['Survived'].loc[output['Survived'] < 0.5] = 0

output['Survived'].loc[output['Survived'] > 0.5] = 1

output['Survived'] = output['Survived'].astype('int32')

print(output)
output.to_csv('submission.csv', index=False)