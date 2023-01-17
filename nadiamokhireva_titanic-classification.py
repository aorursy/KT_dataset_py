import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error, accuracy_score

from tensorflow.keras.layers import Input, Dense, Activation,Dropout

from tensorflow.keras.models import Model
from google.colab import files

import io



# Prompting to upload the train.csv

uploaded_train = files.upload()

df = pd.read_csv(io.BytesIO(uploaded_train['train.csv']), index_col='PassengerId')
# Prompting to upload the test.csv file

uploaded_test = files.upload()
# Save the testing data into a dataframe

df_test = pd.read_csv(io.BytesIO(uploaded_test['test.csv']), index_col='PassengerId')
df.columns
df.dtypes.unique()
df['Family'] = df['SibSp']+df['Parch']
df = df.drop(columns=['SibSp', 'Parch'])
df.isna().sum()
df = df.drop(columns=['Cabin'])
df.groupby(['Pclass'])['Age'].mean()
df['Age'] = df['Age'].fillna(df.groupby(['Pclass'])['Age'].transform(np.mean))
df['Embarked'] = df['Embarked'].fillna('S')
df.isna().sum()
sex_replace = {'female': 0, 'male': 1}

df = df.replace({'Sex': sex_replace})
# Create features and labels

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Family']]

y = df[['Survived']]
X_scaled = scale(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=101)
print(X_train.shape)

print(y_train.shape)
input_layer = Input(shape=(X_train.shape[1]))

output = Dense(y_train.shape[1], activation='sigmoid')(input_layer)



model = Model(inputs=input_layer, outputs=output)



from keras import layers, models

model.compile(optimizer='rmsprop', loss='mse', metrics=['acc'])
print(logistic_regression_model.summary())
history = model.fit(X_train, y_train, batch_size=8, epochs=30, validation_split=0.2)
score = model.evaluate(X_val, y_val)



print("Test Score:", score[0])

print("Test Accuracy:", score[1])
input_layer = Input(shape=(X_train.shape[1]))

layer_1 = Dense(15, activation='relu')(input_layer)

layer_2 = Dense(10, activation='relu')(layer_1)

output = Dense(y_train.shape[1], activation='tanh')(layer_2)



neural_net_model = Model(inputs=input_layer, outputs=output)

neural_net_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print(neural_net_model.summary())
history = neural_net_model.fit(X_train, y_train, batch_size=8, epochs=50, verbose=1, validation_split=0.2)
score = neural_net_model.evaluate(X_val, y_val, verbose=1)



print("Test Score:", score[0])

print("Test Accuracy:", score[1])
decision_tree_model = DecisionTreeClassifier(random_state=101, max_leaf_nodes=50)

decision_tree_model.fit(X_train, y_train)

predict_tree = decision_tree_model.predict(X_val)

accuracy = accuracy_score(y_val, predict_tree)

print('Test accuracy: {:.4f}'.format(accuracy))
df_test['Family'] = df_test['SibSp']+df_test['Parch']

df_test['Age'] = df_test['Age'].fillna(df_test.groupby(['Pclass'])['Age'].transform(np.mean))

df_test['Embarked'] = df_test['Embarked'].fillna('S')

df_test['Fare'] = df_test['Fare'].fillna(df_test.groupby(['Pclass', 'Embarked'])['Fare'].transform(np.mean))

sex_replace = {'female': 0, 'male': 1}

df_test = df_test.replace({'Sex': sex_replace})

X_test = df_test[['Pclass', 'Sex', 'Age', 'Fare', 'Family']]

X_test_scaled = scale(X_test)
test_predict = decision_tree_model.predict(X_test_scaled)
X_test.index
import pandas as pd

test = pd.read_csv("../input/titanic/test.csv")