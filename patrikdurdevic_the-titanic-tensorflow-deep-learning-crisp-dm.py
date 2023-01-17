import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf # Deep learning

import matplotlib.pyplot as plt # Plots

import math # Basic math

%matplotlib inline

import seaborn as sns # Plots



data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

data.describe()
sns.countplot(x="Survived", data=data)
sns.countplot(x="Pclass", hue="Survived", data=data)
data.Name
data.Name.values[240:250]
sns.countplot(x="Sex", hue="Survived", data=data)
sns.kdeplot(data=data.Age, shade=True)
sns.kdeplot(data=data.loc[data['Survived'] == 1].Age, label="Survived", shade=True)

sns.kdeplot(data=data.loc[data['Survived'] == 0].Age, label="Died", shade=True)
sns.regplot(x=data['Age'], y=data['Survived'])
sns.lmplot(x="Age", y="Survived", hue="Sex", data=data)
data.SibSp.describe()
data.Parch.describe()
sns.lmplot(x="Fare", y="Survived", data=data)
data.Cabin.describe()
data.Cabin.unique()
columnFilter = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]

filteredData = data[columnFilter]

filteredTestData = test_data[columnFilter[1:]]

filteredData.head()
filteredData.head()
def oneHotEncode(dataToEncode, column):

    onehot = pd.get_dummies(dataToEncode[column])

    dataToEncode = dataToEncode.join(onehot)

    dataToEncode = dataToEncode.drop(columns=column)

    return dataToEncode



filteredData = oneHotEncode(filteredData, "Pclass")

filteredData.head()
def sex_to_numerical(d):

    sex = d["Sex"]

    if sex == "male":

        return 1

    return 0



filteredData['Sex'] = filteredData.apply(sex_to_numerical, axis=1)

filteredData.head()
from sklearn.preprocessing import MinMaxScaler



def normalize_age(dataToNormalize):

    scaler = MinMaxScaler()

    dataToNormalize["Age"] = scaler.fit_transform(dataToNormalize["Age"].values.reshape(-1,1))

    return dataToNormalize



filteredData.Age.fillna((filteredData['Age'].mean()), inplace=True)



filteredData = normalize_age(filteredData)

filteredData.head()
from sklearn.model_selection import train_test_split

X = filteredData[filteredData.columns[1:]]

y = filteredData[filteredData.columns[0]]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



X_train = X_train.to_numpy()

X_test = X_test.to_numpy()

y_train = y_train.to_numpy()

y_test = y_test.to_numpy()
model = tf.keras.Sequential([

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)



plt.figure(figsize=[8,6])

plt.plot(history.history['accuracy'],'r',linewidth=3.0)

plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)

plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)

plt.xlabel('Epochs ',fontsize=16)

plt.ylabel('Accuracy',fontsize=16)

plt.title('Accuracy Curves',fontsize=16)

plt.show()
filteredTestData = oneHotEncode(filteredTestData, "Pclass")

filteredTestData['Sex'] = filteredTestData.apply(sex_to_numerical, axis=1)

filteredTestData.Age.fillna((filteredTestData['Age'].mean()), inplace=True)

filteredTestData = normalize_age(filteredTestData)
model.fit(X, y, epochs=100)
predictions = model.predict(filteredTestData)

predictions = np.where(predictions > 0.5, 1, 0)

predictions
predictions = pd.DataFrame(predictions).rename(columns={0: "Survived"})

predictions.index.name ="PassengerId"

predictions.index += 892

predictions.to_csv("predictions.csv")

predictions