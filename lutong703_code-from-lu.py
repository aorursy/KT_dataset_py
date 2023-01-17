import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import pydot

import re



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasClassifier



import sklearn

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz, DecisionTreeClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, train_test_split



import os

print(os.listdir("../input"))
data_raw = pd.read_csv('../input/train.csv')

data_val = pd.read_csv('../input/test.csv')

data_raw.head()
data_raw.info()
data_raw.drop(['PassengerId'], 1).hist(bins=50, figsize=(20,15))

plt.show()
def preprocess_data(df):

    

    processed_df = df

    

    # Drop useless columns

    processed_df = processed_df.drop(['PassengerId'], 1)

    

    # Deal with missing values

    processed_df['Embarked'].fillna('S', inplace=True)

    processed_df['Age'].fillna(processed_df['Age'].mean(), inplace=True)

    processed_df['Age'] = processed_df['Age'].astype(int)

    processed_df['Fare'] = processed_df['Fare'].interpolate()

    processed_df['Cabin'].fillna('U', inplace=True)

    

    # feature engineering on columns

    processed_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in data_raw['Name']), index=data_raw.index)

    processed_df['Title'] = processed_df['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    processed_df['Title'] = processed_df['Title'].replace(['Mlle', 'Ms'], 'Miss')

    processed_df['Title'] = processed_df['Title'].replace('Mme', 'Mrs')

    processed_df['Title'] = processed_df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})

    

    processed_df['Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})

    processed_df['Embarked'] = processed_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    

    processed_df['Familly_size'] = processed_df['SibSp'] + processed_df['Parch'] + 1

    processed_df['IsAlone'] = np.where(processed_df['Familly_size']!=1, 0, 1)

    

    #Creation of a deck column corresponding to the letter contained in the cabin value

    processed_df['Cabin'] = processed_df['Cabin'].str[:1]

    processed_df['Cabin'] = processed_df['Cabin'].map({cabin: p for p, cabin in enumerate(set(cab for cab in processed_df['Cabin']))})

    

    processed_df = processed_df.drop(['Name', 'Ticket'], 1)    

    

    return processed_df
data_train = data_raw.copy()

X = data_train.drop(['Survived'], 1)

Y = data_train['Survived']

X = preprocess_data(X)

sc = StandardScaler()

X = pd.DataFrame(sc.fit_transform(X.values), index=X.index, columns=X.columns)

    

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def build_ann(optimizer='adam'):

    

    # Initializing our ANN

    ann = Sequential()

    

    # Adding the input layer and the first hidden layer of our ANN with dropout

    ann.add(Dense(units=32, kernel_initializer='glorot_normal', activation='relu', input_shape = (11,)))

    # Dropout will disable some neurons (here 50% of all neurons) to avoid overfitting

    ann.add(Dropout(p=0.5))

    

    # Add other layers, it is not necessary to pass the shape because there is a layer before

    ann.add(Dense(units=64, kernel_initializer='glorot_normal', activation='relu'))

    ann.add(Dropout(p=0.5))

    ann.add(Dense(units=128, kernel_initializer='glorot_normal', activation='relu'))

    ann.add(Dropout(p=0.5))

    ann.add(Dense(units=164, kernel_initializer='glorot_normal', activation='relu'))

    ann.add(Dropout(p=0.5))

    ann.add(Dense(units=16, kernel_initializer='glorot_normal', activation='relu'))

    

    # Adding the output layer

    ann.add(Dense(units=1, kernel_initializer='glorot_normal', activation='sigmoid'))

    

    # Compilling the ANN

    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    

    return ann
ann = build_ann()

# Training the ANN

ann.fit(X_train, Y_train, batch_size=10, epochs=100)
ann_prediction = ann.predict(X_test)

ann_prediction = (ann_prediction > 0.5) # convert probabilities to binary output



# Compute error between predicted data and true response and display it in confusion matrix

score = metrics.accuracy_score(Y_test, ann_prediction)

print(score)
val = data_val.copy()

val = preprocess_data(val)

val = pd.DataFrame(sc.fit_transform(val.values), index=val.index, columns=val.columns)
prediction = ann.predict(val)

prediction = (prediction > 0.5)*1



result_df = data_val.copy()

result_df['Survived'] = prediction

result_df.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)
result_df[['PassengerId', 'Survived']].head()