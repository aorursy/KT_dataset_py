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
train_df = pd.read_csv('../input/titanic/train.csv')
train_df.info()
train_df.loc[0]
#Drop Name and Ticket



train_df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1, inplace=True)

train_df.info()
X = train_df.drop('Survived', axis=1, inplace=False)

y = train_df['Survived']



print(f'Input features shape: {X.shape}')

print(f'Labels shape: {y.shape}')
#Import visualization libraries



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
sns.barplot(x = y.unique(), y = y.value_counts())

plt.ylabel('Number of Passengers')
frequencies = X.count()



freq_df = pd.DataFrame([frequencies], columns=X.columns)

print(freq_df.head())



plt.figure(figsize=(10, 6))

sns.barplot(x=X.columns, y = frequencies)

plt.ylabel('Frequency')

plt.xlabel('Feature')

plt.show()



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



'''

The following pipeline is only for numerical features:

1. Fill in Missing Values with Median value of existing numbers

2. Scale all the numbers to be in between 0 and 1

'''



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy="median")), 

    ('std_scaler', StandardScaler())

])



'''

The following pipeline is only for categorical features:

1. Fill in Missing Values with Most Frequent value from existing data

2. One-hot encode (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

'''



cat_pipeline = Pipeline([

    ('cat_imputer', SimpleImputer(strategy='most_frequent')),

    ('one_hot', OneHotEncoder())

])

#Combine Pipelines



from sklearn.compose import ColumnTransformer



num_attribs = ['Age', 'SibSp', 'Parch', 'Fare']

cat_attribs = ['Pclass', 'Sex', 'Embarked']



full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), 

                                   ("cat", cat_pipeline, cat_attribs)])

X = full_pipeline.fit_transform(X)

X.shape
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#Import necessary libraries



import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras import layers as L

#Step 1

model = Sequential(name='titanic_model')



#Step 2

model.add(L.InputLayer(input_shape=(12,))) # necessary to use model.summary()



#Step 3

model.add(L.Dense(512, activation='relu'))

model.add(L.Dense(1024, activation='relu'))

model.add(L.Dropout(0.4)) #prevents overfitting by setting 40% of nuerons to 0

model.add(L.Dense(512, activation='relu'))

model.add(L.Dropout(0.4))

model.add(L.Dense(128, activation='relu'))

model.add(L.Dense(64, activation='relu'))

model.add(L.Dense(64, activation='relu'))

model.add(L.Dense(1, activation='sigmoid')) # output layer, use sigmoid for binary



model.summary()
#Step 4

model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['accuracy'])
'''

This custom callback stops training once the validation accuracy reaches 83%.

There are several callbacks already implemented by keras.

https://www.tensorflow.org/api_docs/python/tf/keras/callbacks



'''



class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('val_accuracy')>0.83):

            print("\nReached 83% accuracy so cancelling training!")

            self.model.stop_training = True





#Step 6

history = model.fit(X_train, y_train, 

                    validation_data=(X_val, y_val), 

                    batch_size=8, 

                    callbacks=[myCallback()],

                    epochs=50)
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(['train', 'val'], loc='upper right')

plt.show()



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend(['train', 'val'], loc='upper right')

plt.show()
test_df = pd.read_csv('../input/titanic/test.csv')

test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

test_df
test_prepared = full_pipeline.transform(test_df)

test_prepared.shape
preds = model.predict(test_prepared)

preds[:5]
submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission['Survived'] = [0 if pred < 0.5 else 1 for pred in preds]

submission.head()
from IPython.display import FileLink





submission.to_csv('submission.csv',index=False)

FileLink(r'submission.csv')