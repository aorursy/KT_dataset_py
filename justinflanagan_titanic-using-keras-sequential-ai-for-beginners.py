# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn import preprocessing

from tensorflow.python import keras

from sklearn.impute import SimpleImputer

import tensorflow as tf

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Dropout
train_data = '/kaggle/input/titanic/train.csv'

test_data = "/kaggle/input/titanic/test.csv"
train_panda = pd.read_csv(train_data, index_col = "PassengerId")

train_panda.head()
test_panda = pd.read_csv(test_data)

test_panda.head()
train_panda.Cabin.fillna("M")
test_panda.Cabin.fillna("M")
train_panda.Cabin = train_panda.Cabin.astype('str')
test_panda.Cabin = test_panda.astype('str')
encoder = preprocessing.LabelEncoder()
cat_features = ["Sex", "Embarked","Cabin"]
train_panda["Embarked"] = train_panda["Embarked"].astype(str)
encoded_train = train_panda[cat_features].apply(encoder.fit_transform)

encoded_test = test_panda[cat_features].apply(encoder.fit_transform)
num_features = ["Survived","Pclass","Age","SibSp","Parch","Fare"]

test_features= ["Pclass","Age","SibSp","Parch","Fare"]
training_data = train_panda[num_features].join(encoded_train)

test_data = test_panda[test_features].join(encoded_test)
training_data.head()
test_data.head()
training_data.isnull().sum()
test_data.isnull().sum()
my_imputer = SimpleImputer()

imputed_train = pd.DataFrame(my_imputer.fit_transform(training_data))

imputed_test_data = pd.DataFrame(my_imputer.fit_transform(test_data))
imputed_train.head()
imputed_train.columns = training_data.columns

imputed_test_data.columns = test_data.columns
imputed_train.head()
#from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()

#normalized_train = scaler.fit_transform(imputed_train)

#normalized_test = scaler.fit_transform(imputed_test_data)
#normalized_train_data = pd.DataFrame(data= normalized_train)
#normalized_train_data.columns = training_data.columns
#normalized_test_data = pd.DataFrame(data = normalized_test)
#normalized_test_data.columns = test_data.columns
#normalized_test_data.head()
#normalized_train_data.head()
X = imputed_train.drop("Survived", axis = 1)

y = imputed_train["Survived"]
X.head()
y.head()
X.shape
tf.random.set_seed(42)
model = Sequential()
model.add(Dense(8, activation = "relu", input_shape = (8,)))
model.add(Dense(5, activation = "relu"))
model.add(Dense(2, activation = "softmax"))
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer = "adam", metrics = ['accuracy'])
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=50, restore_best_weights = True, verbose = 1)
model.fit(X, y,

          batch_size=1,

          epochs=1000,

          callbacks = [callback],

          validation_split = 0.2,

          verbose = 1)
preds = model.predict(imputed_test_data)
print(preds)
predictions = np.array(preds).argmax(axis=1)
print(predictions)
passenger_id=test_panda["PassengerId"]

results=passenger_id.to_frame()

results["Survived"]=predictions
results.head()
results.to_csv("Titanic_ai_model.csv", index=False)