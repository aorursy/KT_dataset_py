# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/titanic_data.csv')

df.shape
df.head()
# Drop useless columns

df = df.drop(['Name', 'Ticket', 'Cabin','Embarked'], axis=1)

# Drop invalid data rows

df = df.dropna()

df.head()
# Convert male = 1 / female = 2

sexConvDict = {"male":1, "female":2}

df['Sex'] = df['Sex'].apply(sexConvDict.get).astype(int)

df.head()
features = ['Sex', 'Parch', 'Pclass', 'Age', 'Fare', 'SibSp']

df[features]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X = scaler.fit_transform(df[features].values)

#X = df[features].values

y = df['Survived'].values



# Keras

y_onehot = pd.get_dummies(df['Survived']).values

y_onehot
from sklearn.model_selection import train_test_split



#Keras

X_train, X_test, y_label_train, y_label_test, y_train, y_test = train_test_split(X, y, y_onehot, random_state=0)
from keras.models import Sequential

from keras.layers import Dense, Activation



model = Sequential()



#3 layers

model.add(Dense(input_dim=len(features), output_dim=50))

model.add(Dense(output_dim=25))

model.add(Dense(output_dim=2))

model.add(Activation("softmax"))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, y_train)
predicted = model.predict_classes(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(predicted,y_label_test)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_label_test, predicted)