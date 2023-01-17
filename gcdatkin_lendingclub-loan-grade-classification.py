import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/lending-club/accepted_2007_to_2018Q4.csv.gz', compression='gzip')
data
# Cutting the data in half to avoid out-of-memory issues



data = data.sample(frac=0.5, axis=0, random_state=42).reset_index(drop=True)
data
data.isna().mean().sort_values()
data = data.drop(data.loc[:, data.isna().mean().sort_values() > 0.3].columns, axis=1)
data
data = data.dropna(axis=0).reset_index(drop=True)
data.isna().sum().sum()
data
unneeded_columns = ['id', 'sub_grade', 'emp_title', 'url', 'title', 'zip_code']
{column: list(data[column].unique()) for column in data.drop(unneeded_columns, axis=1).columns if data.dtypes[column] == 'object'}
data = data.drop(unneeded_columns, axis=1)
date_columns = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
data['issue_d']
data.loc[0, 'issue_d'][0:3]
data.loc[0, 'issue_d'][-4:]
for column in date_columns:

    data[column + '_month'] = data[column].apply(lambda x: x[0:3])

    data[column + '_year'] = data[column].apply(lambda x: x[-4:])
data
data = data.drop(date_columns, axis=1)
data
month_ordering = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for column in date_columns:

    data[column + '_month'] = data[column + '_month'].apply(lambda x: month_ordering.index(x))
data
for column in data.columns:

    try:

        data[column] = data[column].astype(np.float)

    except:

        pass
{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
target = 'grade'





binary_features = ['term', 'pymnt_plan', 'initial_list_status', 'application_type', 'hardship_flag', 'disbursement_method', 'debt_settlement_flag']

binary_positives = [' 60 months', 'y', 'w', 'Individual', 'Y', 'Cash', 'Y']



ordinal_features = ['emp_length']

emp_ordering = [

    '< 1 year',

    '1 year',

    '2 years',

    '3 years',

    '4 years',

    '5 years',

    '6 years',

    '7 years',

    '8 years',

    '9 years',

    '10+ years'

]



nominal_features = ['home_ownership', 'verification_status', 'loan_status', 'purpose', 'addr_state']
# Encoding functions



def binary_encode(df, column, positive_value):

    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)



def ordinal_encode(df, column, ordering):

    df[column] = df[column].apply(lambda x: ordering.index(x))



def onehot_encode(df, column):

    dummies = pd.get_dummies(df[column])

    df_new = pd.concat([df, dummies], axis=1)

    df_new = df_new.drop(column, axis=1)

    return df_new
# Perform encoding



for feature, positive_value in zip(binary_features, binary_positives):

    binary_encode(data, feature, positive_value)



ordinal_encode(data, 'emp_length', emp_ordering)



for feature in nominal_features:

    data = onehot_encode(data, feature)
(data.dtypes == 'object').sum()
data[target].value_counts()
# Encoding label column



label_encoder = LabelEncoder()



data[target] = label_encoder.fit_transform(data[target])



target_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}

target_mappings
data
y = data['grade']

X = data.drop('grade', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X.shape
inputs = tf.keras.Input(shape=(166,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(7, activation='softmax')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 20



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()]

)
plt.figure(figsize=(14, 10))



epochs_range = range(epochs)

train_loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Loss Over Time")

plt.legend()



plt.show()
model.evaluate(X_test, y_test)