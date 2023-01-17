# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_orig_df = pd.read_csv('../input/train.csv')

test_orig_df = pd.read_csv('../input/test.csv')

print('\nTrain df:\n', train_orig_df.head())

print('\nTest df:\n', test_orig_df.head())
# Get relevant features in dataframe

train_df = train_orig_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]

test_df = test_orig_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]



print('\nTrain dataframe: \n', train_df.head())

print('\nTest dataframe: \n', test_df.head())

# Fill NaN if any

print('------------------------------')

print('Before fill NaN')

print('train_df null:\n', pd.isnull(train_df).sum())

print('test_df null:\n', pd.isnull(test_df).sum())



# Fill NA train df - Age, Embarked

train_df.Age.fillna(train_df.Age.dropna().max(),inplace =True)

train_df.Embarked.fillna(train_df.Embarked.dropna().max(),inplace =True)



# Fill NA test df - Age, Embarked

test_df.Age.fillna(test_df.Age.dropna().max(),inplace =True)

test_df.Embarked.fillna(test_df.Embarked.dropna().max(),inplace =True)



print('------------------------------')

print('After fill NaN')

print('\ntrain_df null:\n', pd.isnull(train_df).sum())

print('\ntest_df null:\n', pd.isnull(test_df).sum())
# Get ndarray from dataframe

train_X = train_df.iloc[:, :-1].values

train_y = train_df.iloc[:, -1].values



test_X = test_df.iloc[:, :].values



print('Filtered Dataframe: \n',train_df.head())

print('X shape:', train_X.shape)

print('y shape:', train_y.shape)



print('\n------- test -------\n')

print('Filtered Dataframe: \n',test_df.head())

print('test_X shape:', test_X.shape)

print('\n------------\n')
# Label Encoder for Gender & Embarked

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_gender = LabelEncoder()

train_X[:, 1] = label_encoder_gender.fit_transform(train_X[:, 1])

test_X[:, 1] = label_encoder_gender.transform(test_X[:, 1])



label_encoder_embarked = LabelEncoder()

train_X[:, 6] = label_encoder_embarked.fit_transform(train_X[:, 6])

test_X[:, 6] = label_encoder_embarked.transform(test_X[:, 6])



print('train: ',train_X[0:5, :])

print('test: ',test_X[0:5, :])
# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2)

print('Shapes: \n', X_train.shape, X_test.shape,y_train.shape, y_test.shape,)
# Fill NaN values - Imputer

from sklearn.preprocessing import Imputer

#print('Null values before imputer:\n', X_train.isnull().sum())

imputer_age = Imputer()

X_train = imputer_age.fit_transform(X_train)

X_test = imputer_age.transform(X_test)

test_X = imputer_age.transform(test_X)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

test_X = sc_X.transform(test_X)

print(X_train[1, :])

print(X_test[1, :])
import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout



output_dim = int(X_train.shape[1] / 2)

input_dim = int(X_train.shape[1])

dropout_prob = 0.05

batch_size = 20

epochs = 100



classifier = Sequential()



# input layer

classifier.add(Dense(units=output_dim, kernel_initializer='uniform', input_dim=input_dim))

#classifier.add(Dropout(p=dropout_prob))



# hidden layer

#classifier.add(Dense(output_dim=output_dim, init='uniform'))

#classifier.add(Dropout(p=dropout_prob))



# hidden layer

classifier.add(Dense(units=output_dim, kernel_initializer='uniform'))

#classifier.add(Dropout(p=dropout_prob))



# output layer

classifier.add(Dense(units=1, kernel_initializer='uniform'))



# Compile graph

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



classifier.summary()



# fit

classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)

y_pred[1:10,:]
from sklearn.metrics import accuracy_score

metrics = accuracy_score(y_test, y_pred)

metrics
test_y_pred = classifier.predict(test_X)

test_y_pred = (test_y_pred > 0.5)

test_y_pred = test_y_pred.astype(int)



print('Sample prediction:', test_y_pred[10:20,:])

print('Survival count: ',np.count_nonzero(test_y_pred == 1))

print('Death count: ',np.count_nonzero(test_y_pred == 0))
submission = pd.DataFrame({ 'PassengerId': test_orig_df['PassengerId'],

                          'Survived': test_y_pred[:,-1]}) 

submission = submission.to_csv("submission.csv", index=False)

submission = pd.read_csv('submission.csv')

print(submission)