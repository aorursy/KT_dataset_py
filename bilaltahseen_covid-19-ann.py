import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix

import seaborn as sns
dataFrame = pd.read_csv('../input/coronavirusdataset/patient.csv')
dataFrame.describe()
# Mean birth_year = 1972 age = 48 Years Old

# 75% = 1987 = 33 Years Old
infection_reason=dataFrame['infection_reason'].value_counts()

print(infection_reason)

print('-------------------')

print(infection_reason.describe())

fig = plt.figure(figsize=(25, 10),dpi=50)

ax = fig.add_axes([0,0,1,1])

ax.bar(infection_reason.keys().to_list(),infection_reason.to_list())

plt.show()
infection_reason=dataFrame['region'].value_counts()

print(infection_reason)

print('-------------------')

print(infection_reason.describe())

fig = plt.figure(figsize=(25, 10))

ax = fig.add_axes([0,0,1,1])

ax.bar(infection_reason.keys().to_list(),infection_reason.to_list())

plt.show()
infection_reason=dataFrame['confirmed_date'].value_counts()

print(infection_reason)

print('-------------------')

print(infection_reason.describe())

fig = plt.figure(figsize=(25, 10))

ax = fig.add_axes([0,0,1,1])

ax.bar(infection_reason.keys().to_list(),infection_reason.to_list())

plt.show()
infection_reason=dataFrame['state'].value_counts()

print(infection_reason)

print('-------------------')

print(infection_reason.describe())

fig = plt.figure(figsize=(10, 10))

ax = fig.add_axes([0,0,1,1])

ax.bar(infection_reason.keys().to_list(),infection_reason.to_list())

plt.show()

print('Deceased % : '+ str(infection_reason['deceased']/len(dataFrame['state'])*100))
columns=dataFrame['infection_reason'].keys().to_list()

encode_infection_reason=pd.get_dummies(dataFrame['infection_reason'],columns=columns)
le = LabelEncoder()
encode_state=le.fit_transform(dataFrame['state'].values)

encode_state
columns=dataFrame['sex'].keys().to_list()

encode_sex=pd.get_dummies(dataFrame['sex'],columns=columns)
columns=dataFrame['confirmed_date'].keys().to_list()

encode_confirmed_date=pd.get_dummies(dataFrame['confirmed_date'],columns=columns)
columns=dataFrame['released_date'].keys().to_list()

encode_released_date=pd.get_dummies(dataFrame['released_date'],columns=columns)
columns=dataFrame['deceased_date'].keys().to_list()

encode_deceased_date=pd.get_dummies(dataFrame['deceased_date'],columns=columns)
columns=dataFrame['region'].keys().to_list()

encode_region=pd.get_dummies(dataFrame['region'],columns=columns)
columns=dataFrame['group'].keys().to_list()

encode_group=pd.get_dummies(dataFrame['group'],columns=columns)
columns=dataFrame['country'].keys().to_list()

encode_country=pd.get_dummies(dataFrame['country'],columns=columns)
frames=[encode_country,encode_sex,encode_group,encode_region,encode_confirmed_date,encode_released_date,encode_deceased_date,encode_infection_reason]
keys=[]

for i in frames:

    keys+=i.keys().to_list()
finalDataFrame = pd.concat(frames,sort=False,ignore_index=False,axis=1)
X_train, X_test, y_train, y_test = train_test_split(finalDataFrame.values, encode_state, test_size=0.33, random_state=1)
X_train.shape
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(512,activation='relu',input_shape=(finalDataFrame.shape[1],)))

model.add(tf.keras.layers.Dense(256,activation='relu'))

model.add(tf.keras.layers.Dense(128,activation='relu'))

model.add(tf.keras.layers.Dense(64,activation='relu'))

model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,epochs=20,batch_size=20,validation_split=0.3,verbose=1)
model.evaluate(X_test,y_test,verbose=0)
y_predicct=model.predict(finalDataFrame.values)
con_mat = tf.math.confusion_matrix(labels=encode_state, predictions=y_predicct).numpy()
classes=list(dataFrame['state'].unique())

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,

                     index = classes, 

                     columns = classes)
figure = plt.figure(figsize=(4, 4))

sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()