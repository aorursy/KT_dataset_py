import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow.keras import activations

from tensorflow.keras.utils import to_categorical

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/titanic/train.csv')

submission = pd.read_csv('/kaggle/input/titanic/test.csv')
data.head()
ax = sns.scatterplot(x='Parch', y='Age', hue='Survived', data=data)
submission.head()
print('Mean fare of survived passengers: {}'.format(data[data.Survived == 0].Fare.mean()))

print('Mean fare of not survived passengers: {}'.format(data[data.Survived == 1].Fare.mean()))
ax = sns.scatterplot(x="Survived", y="Fare",

                sizes=(1, 8), linewidth=0,

                data=data)
ax = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=data, height=6, kind="bar", palette="muted")
le = LabelEncoder()

le.fit(data.Sex)

data['Sex'] = pd.Series(le.transform(data['Sex'].values))
data['Age'].fillna(value=data['Age'].mean(), inplace=True)
has_age = data[data['Age'].notnull()]
sns.set(rc={'figure.figsize':(9.7,6.27)})

ax = sns.heatmap(has_age.corr(), annot=True, linewidths=.25)
regression = LinearRegression().fit(has_age[['Fare', 'Pclass', 'SibSp', 'Parch']], has_age['Age'])

data.loc[data.Age.isnull(), 'Age'] = data.apply(lambda x: regression.predict([x[['Fare', 'Pclass', 'SibSp', 'Parch']].values])[0], axis=1)
features_list = ['Sex', 'Fare', 'Pclass', 'SibSp', 'Parch']

X_train, X_test, y_train, y_test = train_test_split(data[features_list], 

                                                    data['Survived'], test_size=0.25, random_state=42)
y_train = to_categorical(y_train)
LEARNING_RATE = 0.005

EPOCHES = 20

BATCH_SIZE = 32
model = tf.keras.Sequential([

    tf.keras.layers.Input(len(features_list),),

    tf.keras.layers.Dense(5, activation=activations.tanh),

    tf.keras.layers.Dense(4, activation=activations.tanh),

    tf.keras.layers.Dense(2, activation=activations.tanh),

    tf.keras.layers.Softmax()

    ])



model.summary()



optimizer_ = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

loss_ = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer_, loss=loss_)
history = model.fit(X_train,

        y_train,

        batch_size=BATCH_SIZE,

        epochs=EPOCHES,)
y_pred = np.argmax(model.predict(X_test), axis=1)

print('The accuracy of model is {:0.2f}%'.format(accuracy_score(y_test, y_pred)*100))
submission['Age'].fillna(value=data['Age'].mean(), inplace=True)

submission['Sex'] = pd.Series(le.transform(submission['Sex'].values))
results = model.predict(submission[features_list])

results[:5]
y_pred = np.argmax(results, axis=1)

y_pred
data = {'PassengerId': submission['PassengerId'], 'Survived': y_pred}

submit = pd.DataFrame(data)

submit.to_csv('submit.csv', index=False)