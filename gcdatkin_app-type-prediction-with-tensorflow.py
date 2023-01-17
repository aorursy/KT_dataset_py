import numpy as np

import pandas as pd

import plotly.express as px



import re

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
data
data.info()
unneeded_columns = ['App', 'Genres', 'Current Ver', 'Android Ver']



data = data.drop(unneeded_columns, axis=1)
data
data.isna().sum()
def get_uniques(df, columns):

    return {column: list(df[column].unique()) for column in columns}



def get_categorical_columns(df):

    return [column for column in df.columns if df.dtypes[column] == 'object']
get_uniques(data, get_categorical_columns(data))
data[data['Category'] == '1.9']
data = data.drop(10472, axis=0).reset_index(drop=True)
data
data['Reviews'] = data['Reviews'].astype(np.float)
(data['Size'] == 'Varies with device').sum()
data['Size'] = data['Size'].apply(lambda x: np.NaN if x == 'Varies with device' else x)



data['Size'] = data['Size'].apply(lambda x: np.float(x.replace('M', '')) * 1e6 if type(x) != float and 'M' in x else x)



data['Size'] = data['Size'].apply(lambda x: np.float(x.replace('k', '')) * 1e3 if type(x) != float and 'k' in x else x)
data['Size'].astype(np.float)
data
data['Price'] = data['Price'].apply(lambda x: np.float(x.replace('$', '')))
data
get_uniques(data, get_categorical_columns(data))
data[data['Type'].isna()]
data = data.drop(9148, axis=0).reset_index(drop=True)
data.isna().sum()
data['Rating'] = data['Rating'].fillna(data['Rating'].mean())



data['Size'] = data['Size'].fillna(data['Size'].mean())
data
get_uniques(data, get_categorical_columns(data))
data['Month'] = data['Last Updated'].apply(lambda x: re.search('^[^\s]+', x).group(0))

data['Year'] = data['Last Updated'].apply(lambda x: np.float(re.search('[^\s]+$', x).group(0)))



data = data.drop('Last Updated', axis=1)
data
label_encoder = LabelEncoder()



data['Category'] = label_encoder.fit_transform(data['Category'])

category_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}

category_mappings
data
def binary_encode(df, column, positive_value):

    df = df.copy()

    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)

    return df



def ordinal_encode(df, column, ordering):

    df = df.copy()

    df[column] = df[column].apply(lambda x: ordering.index(x))

    return df
get_uniques(data, get_categorical_columns(data))
installs_ordering = [

    '0+',

    '1+',

    '5+',

    '10+',

    '50+',

    '100+',

    '500+',

    '1,000+',

    '5,000+',

    '10,000+',

    '50,000+',

    '100,000+',

    '500,000+',

    '1,000,000+',

    '5,000,000+',

    '10,000,000+',

    '50,000,000+',

    '100,000,000+',

    '500,000,000+',

    '1,000,000,000+'

]



rating_ordering = [

    'Everyone',

    'Everyone 10+',

    'Teen',

    'Mature 17+',

    'Adults only 18+',

    'Unrated'

]



month_ordering = [

    'January',

    'February',

    'March',

    'April',

    'May',

    'June',

    'July',

    'August',

    'September',

    'October',

    'November',

    'December'

]
data = binary_encode(data, 'Type', 'Paid')



data = ordinal_encode(data, 'Installs', installs_ordering)

data = ordinal_encode(data, 'Content Rating', rating_ordering)

data = ordinal_encode(data, 'Month', month_ordering)
data
y = data['Category']

X = data.drop('Category', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
X.shape
inputs = tf.keras.Input(shape=(9,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(33, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 64

epochs = 100



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

    history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Training and Validation Loss"

)



fig.show()
model.evaluate(X_test, y_test)
1 / 33