print("Hello Capstone Project Course!")
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import MarkerCluster

import warnings

warnings.filterwarnings("ignore")

sns.set()
!wget -O data.csv "https://opendata.arcgis.com/datasets/5b5c745e0f1f48e7a53acec63a0022ab_0.csv"
data = pd.read_csv("data.csv")

data.info()
map = folium.Map(location=[47.60, -122.33], zoom_start=12)

marker_cluster = MarkerCluster().add_to(map)

locations = data[['Y', 'X']][data['Y'].notna()].head(1000)

locationlist = locations.values.tolist()

for point in range(len(locations)):

    folium.Marker(locationlist[point]).add_to(marker_cluster)

map
data['WEATHER'].value_counts().to_frame('count')
data['ROADCOND'].value_counts().to_frame('count')
data['LIGHTCOND'].value_counts().to_frame('count')
data['SPEEDING'].value_counts().to_frame()
data['SEVERITYCODE'].value_counts().to_frame('count')
data['UNDERINFL'].value_counts().to_frame('count')
data['PERSONCOUNT'].describe()
data['VEHCOUNT'].describe()
data['PEDCOUNT'].describe()
data['PEDCYLCOUNT'].describe()
data.isna().sum()
data.duplicated().sum()
data_clean = data[['X', 'Y', 'WEATHER', 'ROADCOND', 'LIGHTCOND',

                   'SPEEDING', 'SEVERITYCODE', 'UNDERINFL',

                   'SERIOUSINJURIES', 'FATALITIES', 'INJURIES',

                   'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT']]

data_clean.info()
data_clean['SPEEDING'] = data_clean['SPEEDING'].map({'Y': 1})

data_clean['SPEEDING'].replace(np.nan, 0, inplace=True)

data_clean['SPEEDING'].value_counts().to_frame()
data_clean.replace('Unknown', np.nan, inplace=True)

data_clean.replace('Other', np.nan, inplace=True)

data_clean['SEVERITYCODE'].replace('0', np.nan, inplace=True)
sns.heatmap(data_clean.isnull(), cmap='YlGnBu_r')

plt.show()
data_clean.dropna(axis=0, inplace=True)
sns.heatmap(data_clean.isnull(), cmap='YlGnBu_r')

plt.show()
data_clean['UNDERINFL'] = data_clean['UNDERINFL'].map({'N': 0, '0': 0, 'Y': 1, '1': 1})
data_clean.info()
ax = sns.countplot(data_clean['WEATHER'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, 

                   horizontalalignment='right')

plt.show()
ax = sns.countplot(data_clean['ROADCOND'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, 

                   horizontalalignment='right')

plt.show()
ax = sns.countplot(data_clean['LIGHTCOND'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, 

                   horizontalalignment='right')

plt.show()
sns.countplot(data_clean['UNDERINFL'])
ax = plt.scatter(data_clean['VEHCOUNT'], data_clean['PERSONCOUNT'])

plt.xlabel('VEHCOUNT')

plt.ylabel('PERSONCOUNT')

plt.show()
ax = plt.scatter(data_clean['VEHCOUNT'], data_clean['INJURIES'])

plt.xlabel('VEHCOUNT')

plt.ylabel('INJURIES')

plt.show()
ax = plt.scatter(data_clean['PEDCOUNT'], data_clean['PERSONCOUNT'])

plt.xlabel('PEDCOUNT')

plt.ylabel('PERSONCOUNT')

plt.show()
sns.heatmap(data_clean.corr(), cmap='YlGnBu_r')

plt.show()
data_clean = pd.concat([data_clean.drop(['WEATHER', 'ROADCOND', 'LIGHTCOND'], axis=1), 

           pd.get_dummies(data_clean['ROADCOND']),

           pd.get_dummies(data_clean['LIGHTCOND']),

           pd.get_dummies(data_clean['WEATHER'])], axis=1)
data_clean = data_clean.sample(frac=1).reset_index(drop=True)
data_clean.head(5).T
sns.heatmap(data_clean.corr(), cmap='YlGnBu_r')

plt.show()
from sklearn import preprocessing

x = data_clean.drop(['SEVERITYCODE'], axis=1)

y = data_clean[['SEVERITYCODE']]

data_clean_scaled = preprocessing.StandardScaler().fit(x).transform(x)

data_clean_scaled[0:3]
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_clean_scaled, y, 

                                                    test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

dTreeModel = DecisionTreeClassifier(criterion='entropy', max_depth=5)

dTreeModel.fit(x_train, y_train)

dTreeModel
yHat = dTreeModel.predict(x_test)
print(classification_report(y_test, yHat))
from sklearn.ensemble import RandomForestClassifier

rfcModel = RandomForestClassifier(n_estimators=75)

rfcModel.fit(x_train, y_train)
yHat = rfcModel.predict(x_test)
print(classification_report(y_test, yHat))
from sklearn.linear_model import LogisticRegression

logRegModel = LogisticRegression(C=0.01)

logRegModel.fit(x_train, y_train)

logRegModel
yHat = logRegModel.predict(x_test)
print(classification_report(y_test, yHat))
import tensorflow as tf



model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(32, input_dim=x_train.shape[1], activation='relu'),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(4, activation='sigmoid')

])



model.compile(

    loss='categorical_crossentropy', 

    optimizer='adam', 

    metrics=['accuracy']

)
num_epochs = 10

history = model.fit(x_train, tf.keras.utils.to_categorical(

    y_train['SEVERITYCODE'].map({

        '1': 0,

        '2': 1,

        '2b': 2,

        '3': 3

    }), dtype='float32'

), epochs=num_epochs, batch_size=50, validation_split = 0.2)
loss_train = history.history['loss']

loss_validation = history.history['val_loss']

epochs = range(1, num_epochs + 1)

plt.plot(epochs, loss_train, 'g', label='Training')

plt.plot(epochs, loss_validation, 'b', label='Validation')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Loss')

plt.legend()

plt.show()
acc_train = history.history['accuracy']

acc_validation = history.history['val_accuracy']

epochs = range(1, num_epochs + 1)

plt.plot(epochs, acc_train, 'g', label='Training')

plt.plot(epochs, acc_validation, 'b', label='Validation')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Accuracy')

plt.legend()

plt.show()
yHat = model.predict(x_test)

yPred = [np.argmax(y) for y in yHat]
print(classification_report(y_test.SEVERITYCODE.map({

        '1': 0,

        '2': 1,

        '2b': 2,

        '3': 3

}), yPred))
plt.bar(['DTC', 'RFC', 'LogReg', 'ANN'], [1.,1.,1.,1.])

plt.show()