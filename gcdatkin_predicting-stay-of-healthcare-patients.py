import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

import tensorflow as tf
data = pd.read_csv('../input/av-healthcare-analytics-ii/healthcare/train_data.csv')
data
data.isnull().sum()
def impute_missing_values(data, columns):

    for column in columns:

        data[column] = data[column].fillna(data[column].mean())
impute_columns = ['Bed Grade', 'City_Code_Patient']



impute_missing_values(data, impute_columns)
data.isnull().sum()
data.dtypes
def get_categorical_uniques(data):

    categorical_columns = [column for column in data.dtypes.index if data.dtypes[column] == 'object']

    categorical_uniques = {column: data[column].unique() for column in categorical_columns}

    

    return categorical_uniques
get_categorical_uniques(data)
pd.get_dummies(data['Department'])
def onehot_encode(data, columns):

    for column in columns:

        dummies = pd.get_dummies(data[column])

        data = pd.concat([data, dummies], axis=1)

        data.drop(column, axis=1, inplace=True)

    return data
onehot_columns = ['Hospital_type_code', 'Hospital_region_code', 'Department', 'Ward_Type', 'Ward_Facility_Code']
data = onehot_encode(data, onehot_columns)
data
categorical_uniques = get_categorical_uniques(data)

get_categorical_uniques(data)
for column in categorical_uniques:

    categorical_uniques[column] = sorted(categorical_uniques[column])
categorical_uniques
unique_list = categorical_uniques['Type of Admission']

unique_list.insert(0, unique_list.pop(unique_list.index('Urgent')))

unique_list.insert(0, unique_list.pop(unique_list.index('Trauma')))



unique_list = categorical_uniques['Severity of Illness']

unique_list.insert(0, unique_list.pop(unique_list.index('Moderate')))

unique_list.insert(0, unique_list.pop(unique_list.index('Minor')))
categorical_uniques
stay_mappings = {value: index for index, value in enumerate(categorical_uniques['Stay'])}

stay_mappings
def ordinal_encode(data, uniques):

    for column in uniques:

        data[column] = data[column].apply(lambda x: uniques[column].index(x))
data['Stay']
ordinal_encode(data, categorical_uniques)

data['Stay']
(data.dtypes == 'object').sum()
data
data.set_index('case_id', inplace=True)
y = data['Stay']

X = data.drop('Stay', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
log_model = LogisticRegression()

log_model.fit(X_train, y_train)
inputs = tf.keras.Input(shape=(38,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(11, activation='softmax')(x)



nn_model = tf.keras.Model(inputs=inputs, outputs=outputs)
nn_model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)
batch_size = 32

epochs = 10



history = nn_model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

)
print(f"Logistic Regression Acc: {log_model.score(X_test, y_test)}")

print(f"     Neural Netowrk Acc: {nn_model.evaluate(X_test, y_test, verbose=0)[1]}")
plt.figure(figsize=(14, 10))



plt.plot(range(epochs), history.history['loss'], label="Training Loss")

plt.plot(range(epochs), history.history['val_loss'], label="Validation Loss")



plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend(loc='upper right')



plt.show()
np.argmin(history.history['val_loss']) + 1