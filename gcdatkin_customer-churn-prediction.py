import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data
data.info()
data = data.drop('customerID', axis=1)
data.isna().sum()
data
def get_uniques(df, columns):

    return {column: list(df[column].unique()) for column in columns}
def get_categorical_columns(df):

    return [column for column in df.columns if df.dtypes[column] == 'object']
get_uniques(data, get_categorical_columns(data))
sorted(data['TotalCharges'].unique())
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.NaN)



data['TotalCharges'] = data['TotalCharges'].astype(np.float)



data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
data['MultipleLines'] = data['MultipleLines'].replace('No phone service', 'No')



data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

      'TechSupport', 'StreamingTV', 'StreamingMovies']] = data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                                                                'TechSupport', 'StreamingTV', 'StreamingMovies']].replace('No internet service', 'No')
get_uniques(data, get_categorical_columns(data))
binary_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',

                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

                   'StreamingTV', 'StreamingMovies', 'PaperlessBilling']



ordinal_features = ['InternetService', 'Contract']



nominal_features = ['PaymentMethod']





target_column = 'Churn'
internet_ordering = ['No', 'DSL', 'Fiber optic']

contract_ordering = ['Month-to-month', 'One year', 'Two year']
def binary_encode(df, column, positive_value):

    df = df.copy()

    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)

    return df



def ordinal_encode(df, column, ordering):

    df = df.copy()

    df[column] = df[column].apply(lambda x: ordering.index(x))

    return df

    

def onehot_encode(df, column):

    df = df.copy()

    dummies = pd.get_dummies(df[column])

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
data = binary_encode(data, 'gender', 'Male')



yes_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines',

                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',

                'StreamingTV', 'StreamingMovies', 'PaperlessBilling']



for feature in yes_features:

    data = binary_encode(data, feature, 'Yes')





data = ordinal_encode(data, 'InternetService', internet_ordering)

data = ordinal_encode(data, 'Contract', contract_ordering)





data = onehot_encode(data, 'PaymentMethod')
data = binary_encode(data, 'Churn', 'Yes')
data
y = data['Churn']

X = data.drop('Churn', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
X.shape
y.sum() / len(y)
inputs = tf.keras.Input(shape=(22,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[tf.keras.metrics.AUC(name='auc')]

)





batch_size = 64

epochs = 5



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
plt.figure(figsize=(14, 10))



epochs_range = range(1, epochs + 1)

train_loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.title("Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.show()
np.argmin(val_loss)
model.evaluate(X_test, y_test)