# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf # tensorflow for the win
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from random import randint
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load the data
raw_data = pd.read_csv('../input/train.csv')
raw_data.loc[:100]
age_survived = raw_data.loc[raw_data['Survived']==1,'Age']
age_dead = raw_data.loc[raw_data['Survived']==0,'Age']
age_survived.hist(color='green',bins=30)
age_dead.hist(alpha=0.5, color='red',bins=30)
plt.title('Age and survival (red = dead, green = alive)')
plt.figure()
raw_data['Age'].hist(alpha=0.5, color='blue',bins=30)
##age_ratio = age_survived/(age_survived+age_dead)
#plt.figure()
surv = age_survived.value_counts()
dead = age_dead.value_counts()
df = pd.DataFrame({'surv':surv,'dead':dead})
df.fillna(0, inplace=True) # replace NaN with 0
df['total'] = df['surv'] + df['dead']
df['ratio'] = df['surv']/df['total']
plt.figure()
plt.title("ratio survived/total by age")
df['ratio'].plot()
mean = df['ratio'].mean()
median = df['ratio'].median()
plt.plot([0,80],[mean, mean])
plt.plot([0,80],[median, median])
# First : slipt the input and the expected output
data = raw_data[['Age','Sex','Pclass','Fare', 'SibSp', 'Parch', 'Embarked']].copy() # select only what I think may be useful
y = raw_data['Survived'].copy()

# Second : deal with missing data
mean_age = raw_data['Age'].mean()
data['Age'].fillna(0,inplace=True)
data['Embarked'].fillna(value=randint(0,2),inplace=True)

# Third : make everything a number
data['Sex'] = data['Sex'].map({'female':0,'male':1})
data['Embarked'] = data['Embarked'].replace({'C':0,'Q':1,'S':2})

# normalize everything
#data=(data-data.mean())/data.std()
# same with da test data
test_data_raw = pd.read_csv('../input/test.csv')
test_data = test_data_raw[['Age','Sex','Pclass','Fare', 'SibSp', 'Parch', 'Embarked']].copy()
test_data['Age'].fillna(0,inplace=True)
test_data['Embarked'].fillna(randint(0, 2),inplace=True)
test_data.loc[[152]] = 30.0 # OMG I don't care, just a random value
# Third : make everything a number
test_data['Sex'] = test_data['Sex'].replace({'female':0,'male':1})
test_data['Embarked'] = test_data['Embarked'].replace({'C':0,'Q':1,'S':2})
# normalize everything
#test_data=(test_data-test_data.mean())/test_data.std()
data.head()
# to spot any null/nan values
print(test_data.isnull().values.any())
np.where(test_data.isnull())[0]
# Now we are ready, let's go
# let's build a random neural network
model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(data.values, y.values, epochs=500, batch_size=32)
tf.keras.models.save_model(
    model,
    '../savingModel.h',
    overwrite=True,
    include_optimizer=True
)
predictions = model.predict(test_data)
predictions = np.where(predictions < 0.5,0,1)
predictions = pd.Series(map(lambda x: x[0], predictions))
# submission
sub = pd.DataFrame({'PassengerId':test_data_raw['PassengerId'],
                    'Survived':predictions})
sub.head()
sub.to_csv('submission.csv', index=False)
