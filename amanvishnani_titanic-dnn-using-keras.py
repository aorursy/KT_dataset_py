# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.head()
def clean_data(df):
    df.Age.fillna(value=df.Age.mean(), inplace=True)
    df.Fare.fillna(value=df.Fare.mean(), inplace=True)
    df.Embarked.fillna(value=df.Embarked.value_counts().idxmax(), inplace=True)
    try:
        df.Survived.fillna(value=-1, inplace=True)
    except:
        print("Test Data found.")
    df.drop(columns=['PassengerId','Name','Ticket','Cabin'], inplace=True)
    df.Embarked = pd.Categorical(df.Embarked)
    df.Sex = pd.Categorical(df.Sex)
    df['Embarkedi'] = df.Embarked.cat.codes;
    df['Sexi'] = df.Sex.cat.codes;
    df.drop(columns=['Embarked', 'Sex'], inplace=True);
clean_data(df)
df.head()
df.shape
trainX, trainY = df.iloc[:,1:], df.iloc[:,0]
trainX.head()
trainY.head()
from keras.models import Sequential
from keras.layers.core import Dense

def get_fc_model():
    model = Sequential();
    model.add(Dense(16, input_shape=(7,), activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
fc_model = get_fc_model()
fc_model.compile(optimizer='Adam', loss='mean_squared_error',
                  metrics=['accuracy'])
fc_model.optimizer.lr=0.01
fc_model.fit(x=trainX.values, y=trainY.values, epochs=100)
dfTestX = pd.read_csv('../input/test.csv')
clean_data(dfTestX)
dfTestX.head()
predictions = fc_model.predict_classes(dfTestX, verbose=0)
predictions= predictions.reshape(1,418)
predictions = predictions[0]
submissions=pd.DataFrame({'PassengerId': list(range(892,len(predictions)+892)), 'Survived': predictions})
submissions.to_csv("gender_submission.csv", index=False, header=True)