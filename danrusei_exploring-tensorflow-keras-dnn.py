## Import libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



from sklearn.preprocessing import StandardScaler



import tensorflow as tf

from tensorflow import keras

tf.__version__
# Load dataset

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



# Adding a column in each dataset before merging

train['Type'] = 'train'

test['Type'] = 'test'



# Merging train and test

data = train.append(test, sort=False)
# explore the initial data, first 5 rows of the dataset

data.head()
# Cleaning name and extracting Title

for name_string in data['Name']:

    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)

    

# Replacing rare titles 

mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 

           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 

           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 

           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}

           

data.replace({'Title': mapping}, inplace=True)
data['Family_Size'] = (data['Parch'] + data['SibSp']).astype(int)
data.info()
data.groupby(["Sex", "Pclass"]).Age.mean()
data['Age'].fillna(data.groupby(["Sex", "Pclass"])['Age'].transform("mean"), inplace=True)
data.loc[pd.isnull(data['Embarked'])]
data.loc[61,'Embarked'] = 'S'

data.loc[829,'Embarked'] = 'S'
data['Fare'].fillna(data['Fare'].mean(), inplace = True)
# First drop the variables we won't be using in the model

data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
# convert to cateogry dtype

data['Sex'] = data['Sex'].astype('category')

# convert to category codes

data['Sex'] = data['Sex'].cat.codes
# subset all categorical variables which need to be encoded

categorical = ['Embarked', 'Title', 'Pclass']



for cat in categorical:

    data = pd.concat([data, 

                    pd.get_dummies(data[cat], prefix=cat)], axis=1)

    del data[cat]
# scale numerical values

continuous = ['Age', 'Fare', 'Family_Size']



scaler = StandardScaler()



for val in continuous:

    data[val] = data[val].astype('float64')

    data[val] = scaler.fit_transform(data[val].values.reshape(-1, 1))
# checkout the data after all transoformations

data.head()
#Generate descriptive statistics. Descriptive statistics include those that summarize the central tendency, 

#dispersion and shape of a dataset’s distribution

data.describe()
# Cutting train and test

train = data[data['Type'] == 'train'].drop(columns = ['Type', 'Survived'])

train_ = data[data['Type'] == 'train']['Survived']



test = data[data['Type'] == 'test'].drop(columns = ['Type', 'Survived'])



X_train = train.values

y_train = train_.values



X_test = test.values

X_test = X_test.astype(np.float64, copy=False)
train.shape
# Simple model

model1 = keras.models.Sequential()

model1.add(keras.layers.Dense(18, input_dim = X_train.shape[1], activation = keras.activations.relu))

model1.add(keras.layers.Dense(8, activation = keras.activations.relu))

model1.add(keras.layers.Dense(1, activation = keras.activations.sigmoid))
# visualize the model

model1.summary()
# further explore the model

weights, biases = model1.layers[1].get_weights()

weights.shape
# Compiling our model

model1.compile(optimizer = keras.optimizers.SGD(), 

               loss = keras.losses.binary_crossentropy, 

               metrics = [tf.keras.metrics.binary_accuracy])
history = model1.fit(X_train, y_train, epochs=100, validation_split=0.2)
val_acc = np.mean(history.history['val_binary_accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
history.params
# Plot the learning curves

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

plt.show()
# calculate predictions, this model scores 0.76555 on Kaggle



submission = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')

submission['Survived'] = model1.predict(X_test)

submission['Survived'] = submission['Survived'].apply(lambda x: round(x,0)).astype('int')

submission.to_csv('Titanic_model1.csv')
input_ = keras.layers.Input(shape=X_train.shape[1:])

hidden1 = keras.layers.Dense(36, activation="elu")(input_)

drop1 = keras.layers.Dropout(rate=0.2)(hidden1)

hidden2 = keras.layers.Dense(18, activation="elu")(drop1)

drop2 = keras.layers.Dropout(rate=0.2)(hidden2)

hidden3 = keras.layers.Dense(8, activation="elu")(drop2)

output = keras.layers.Dense(1, activation="sigmoid")(hidden3)

model2 = keras.models.Model(inputs=[input_], outputs=[output])
model2.summary()
# Compile the model

model2.compile(optimizer = 'nadam', 

               loss = 'binary_crossentropy', 

               metrics = ['accuracy'])
# configure EarlyStopping and ask to restore the best weights

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
history2 = model2.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping_cb])
val_acc = np.mean(history2.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
# Plot the learning curves

pd.DataFrame(history2.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

plt.show()
submission2 = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')

submission2['Survived'] = model2.predict(X_test)

submission2['Survived'] = submission['Survived'].apply(lambda x: round(x,0)).astype('int')

submission2.to_csv('Titanic_model2.csv')
def build_model(input_shape=[16], n_hidden=1, n_neurons=30, activation = 'relu', optimizer = 'SGD'):

    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))

    i = 1

    for layer in range(n_hidden):

        model.add(keras.layers.Dense(n_neurons/i, activation=activation))

        if n_neurons > 20:

            model.add(keras.layers.Dropout(rate=0.2))

            i = i + 2

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


model3 = keras.wrappers.scikit_learn.KerasClassifier(build_fn=build_model)
history3 = model3.fit(X_train, y_train, epochs=100,

              validation_split=0.2,

              callbacks=[keras.callbacks.EarlyStopping(patience=10)])
val_acc = np.mean(history3.history['val_accuracy'])

print("\n%s: %.2f%%" % ('val_acc', val_acc*100))
from sklearn.model_selection import RandomizedSearchCV



param_distribs = {

    "n_hidden": [1, 2, 3, 4],

    "n_neurons": [6, 18, 30, 42, 56, 77, 84, 100],

    "activation": ['relu', 'selu', 'elu'],

    "optimizer": ['SGD', 'RMSprop', 'Adam'],

}



rnd_search_cv = RandomizedSearchCV(model3, param_distribs, n_iter=10, cv=3, verbose=2)

rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split=0.2)
rnd_search_cv.best_params_
model4 = build_model(n_hidden=4, n_neurons=77, input_shape=[16], activation = 'elu', optimizer = 'Adam')



print(model4.summary())
history4 = model4.fit(X_train, y_train, epochs=100,

                     validation_split=0.2, callbacks=[early_stopping_cb])
# Plot the learning curves

pd.DataFrame(history4.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1)

plt.show()
submission4 = pd.read_csv("../input/titanic/gender_submission.csv", index_col='PassengerId')

submission4['Survived'] = model4.predict(X_test)

submission4['Survived'] = submission['Survived'].apply(lambda x: round(x,0)).astype('int')

submission4.to_csv('Titanic_model4.csv')