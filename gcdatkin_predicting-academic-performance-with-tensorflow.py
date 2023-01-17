import numpy as np

import pandas as pd

import plotly.express as px



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/xAPI-Edu-Data/xAPI-Edu-Data.csv')
data
data.info()
data.isna().sum()
def get_uniques(df, columns):

    return {column: list(df[column].unique()) for column in columns}



def get_categorical_columns(df):

    return [column for column in df.columns if df.dtypes[column] == 'object']
get_uniques(data, get_categorical_columns(data))
binary_features = ['gender', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']



ordinal_features = ['StageID', 'GradeID']



nominal_features = ['NationalITy', 'PlaceofBirth', 'SectionID', 'Topic']





target_column = 'Class'
binary_positive_values = ['M', 'S', 'Father', 'Yes', 'Good', 'Above-7']



stage_ordering = ['lowerlevel', 'MiddleSchool', 'HighSchool']

grade_ordering = [

    'G-02',

    'G-04',

    'G-05',

    'G-06',

    'G-07',

    'G-08',

    'G-09',

    'G-10',

    'G-11',

    'G-12'

]



nominal_prefixes = ['N', 'B', 'S', 'T']
def binary_encode(df, column, positive_value):

    df = df.copy()

    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)

    return df



def ordinal_encode(df, column, ordering):

    df = df.copy()

    df[column] = df[column].apply(lambda x: ordering.index(x))

    return df



def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
for feature, positive_value in zip(binary_features, binary_positive_values):

    data = binary_encode(data, feature, positive_value)
data = ordinal_encode(data, 'StageID', stage_ordering)

data = ordinal_encode(data, 'GradeID', grade_ordering)
for feature, prefix in zip(nominal_features, nominal_prefixes):

    data = onehot_encode(data, feature, prefix)
target_ordering = ['L', 'M', 'H']

data = ordinal_encode(data, target_column, target_ordering)
data
y = data[target_column]

X = data.drop(target_column, axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
X.shape
inputs = tf.keras.Input(shape=(55,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(3, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 64

epochs = 16



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
fig = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Training and Validation Loss"

)



fig.show()
np.argmin(history.history['val_loss'])
model.evaluate(X_test, y_test)