import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/nys-environmental-remediation-sites/environmental-remediation-sites.csv')
data
data.info()
data.isna().sum()
null_columns = data.loc[:, data.isna().sum() > 0.25 * data.shape[0]]



data = data.drop(null_columns, axis=1)
data
data.isna().sum()
unneeded_columns = ['Program Number', 'Project Name', 'Program Facility Name', 'Address1',

                    'Locality', 'ZIPCode', 'SWIS Code', 'Owner Name', 'Owner Address1',

                    'Owner City', 'Owner State', 'Owner ZIP', 'Location 1']



data = data.drop(unneeded_columns, axis=1)
data
def get_uniques(df, columns):

    return {column: list(df[column].unique()) for column in columns}



def get_categorical_columns(df):

    return [column for column in df.columns if df.dtypes[column] == 'object']
get_uniques(data, get_categorical_columns(data))
data['Project Completion Date']
data['Project Completion Date'] = data['Project Completion Date'].apply(lambda x: x[0:7] if str(x) != 'nan' else x)



data['Year'] = data['Project Completion Date'].apply(lambda x: np.float(x[0:4]) if str(x) != 'nan' else x)

data['Month'] = data['Project Completion Date'].apply(lambda x: np.float(x[5:7]) if str(x) != 'nan' else x)



data = data.drop('Project Completion Date', axis=1)
data
data.isna().sum()
for column in ['New York Zip Codes', 'Counties', 'Year', 'Month']:

    data[column] = data[column].fillna(data[column].mean())
data.isna().sum()
data
def onehot_encode(df, column):

    df = df.copy()

    dummies = pd.get_dummies(df[column])

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
nominal_features = get_categorical_columns(data)

nominal_features.remove('Program Type')



for feature in nominal_features:

    data = onehot_encode(data, feature)
data
(data.dtypes == 'object').sum()
label_encoder = LabelEncoder()



data['Program Type'] = label_encoder.fit_transform(data['Program Type'])



target_mappings = {index: column for index, column in enumerate(label_encoder.classes_)}

target_mappings
y = data['Program Type']

X = data.drop('Program Type', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
X.shape
y.value_counts()
inputs = tf.keras.Input(shape=(117,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(5, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 64

epochs = 60



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Training and Validation Loss"

)



fig.show()
model.evaluate(X_test, y_test)
for label in range(5):

    label_indices = y_test[y_test == label].index

    label_acc = model.evaluate(X_test.loc[label_indices, :], y_test.loc[label_indices], verbose=0)

    print(f"Class {label} Accuracy: {label_acc[1]}")