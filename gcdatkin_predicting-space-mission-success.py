import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import re



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
data
data.drop([data.columns[0], data.columns[1], 'Location', 'Detail'], axis=1, inplace=True)
data
data.columns
data.columns = ['Company Name', 'Datum', 'Status Rocket', 'Rocket', 'Status Mission']
data.isnull().sum()
data['Rocket'].unique()
for value in data['Rocket']:

    print(type(value))
data['Rocket'] = data['Rocket'].astype(str).apply(lambda x: x.replace(',', '')).astype(np.float32)
data['Rocket'] = data['Rocket'].fillna(data['Rocket'].mean())
data.isnull().sum()
data
def get_year_from_date(date):

    year = re.search(r'[^,]*$', date).group(0)

    year = re.search(r'^\s[^\s]*', year).group(0)

    return np.int16(year)
def get_month_from_date(date):

    month = re.search(r'^[^0-9]*', date).group(0)

    month = re.search(r'\s.*$', month).group(0)

    return month.strip()

    
data['Year'] = data['Datum'].apply(get_year_from_date)

data['Month'] = data['Datum'].apply(get_month_from_date)

data.drop('Datum', axis=1, inplace=True)
data
data['Status Mission'].unique()
data['Status Mission'] = data['Status Mission'].apply(lambda x: x if x == 'Success' else 'Failure')
encoder = LabelEncoder()



data['Status Mission'] = encoder.fit_transform(data['Status Mission'])
data
month_ordering = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
data['Status Rocket'].unique()
status_ordering = ['StatusRetired', 'StatusActive']
# Given some data, a column of that data, and an ordering of the values in that column,

# perform ordinal encoding on the column and return the result.



def ordinal_encode(data, column, ordering):

    return data[column].apply(lambda x: ordering.index(x))
data['Month'] = ordinal_encode(data, 'Month', month_ordering)

data['Status Rocket'] = ordinal_encode(data, 'Status Rocket', status_ordering)
data
def onehot_encode(data, column):

    dummies = pd.get_dummies(data[column])

    data = pd.concat([data, dummies], axis=1)

    data.drop(column, axis=1, inplace=True)

    return data
data = onehot_encode(data, 'Company Name')
data
y = data['Status Mission']

X = data.drop('Status Mission', axis=1)
scaler = MinMaxScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
y.sum() / len(y)
inputs = tf.keras.Input(shape=(60,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[tf.keras.metrics.AUC(name='auc')]

)





batch_size=32

epochs=35



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs

)
plt.figure(figsize=(14, 10))



epochs_range = range(1, epochs + 1)

train_loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend('upper right')



plt.show()
np.argmin(val_loss)
model.evaluate(X_test, y_test)