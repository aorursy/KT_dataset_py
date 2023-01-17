import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf



from sklearn.decomposition import PCA
data = pd.read_csv('../input/pokemon/Pokemon.csv')
data
data_raw = data.copy()
data.info()
data.isna().sum()
data = data.drop(['#', 'Name', 'Type 2'], axis=1)
data['Legendary'] = data['Legendary'].astype(np.int)
data
data['Type 1'].unique()
numeric_columns = data.drop('Type 1', axis=1).columns
correlation_matrix = data[numeric_columns].corr()



plt.figure(figsize=(18, 15))

sns.heatmap(correlation_matrix, annot=True, vmin=-1.0, vmax=1.0)

plt.show()
plt.figure(figsize=(20, 10))

for column in ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:

    sns.kdeplot(data[column], shade=True)

plt.show()
data.dtypes
def onehot_encode(df, column, prefix):

    df = df.copy()

    dummies = pd.get_dummies(df[column], prefix=prefix)

    df = pd.concat([df, dummies], axis=1)

    df = df.drop(column, axis=1)

    return df
data = onehot_encode(data, 'Type 1', 't')
data
y = data['Legendary']

X = data.drop('Legendary', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X.shape
inputs = tf.keras.Input(shape=(26,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc')

    ]

)





batch_size = 32

epochs = 20



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
fig_loss = px.line(

    history.history,

    y=['loss', 'val_loss'],

    labels={'x': "Epoch", 'y':"Loss"},

    title="Loss Over Time"

)



fig_loss.show()
np.argmin(history.history['val_loss'])
fig_auc = px.line(

    history.history,

    y=['auc', 'val_auc'],

    labels={'x': "Epoch", 'y':"AUC"},

    title="AUC Over Time"

)



fig_auc.show()
model.evaluate(X_test, y_test)
predictions = np.hstack((model.predict(X_test) >= 0.5).astype(np.int)) != y_test

predictions
mislabeled_indices = y_test[predictions].index
data_raw.loc[mislabeled_indices, :]
X.shape
pca = PCA(n_components=2)

data_reduced = pd.DataFrame(pca.fit_transform(data), columns=["PC1", "PC2"])
data_reduced
legendary_indices = data.query("Legendary == 1").index



mislabeled_legendary_indices = np.intersect1d(mislabeled_indices, legendary_indices)
plt.figure(figsize=(20, 10))



plt.scatter(data_reduced['PC1'], data_reduced['PC2'], c='lightgray')

plt.scatter(data_reduced.loc[legendary_indices, 'PC1'], data_reduced.loc[legendary_indices, 'PC2'], c='dimgray')

plt.scatter(data_reduced.loc[mislabeled_indices, 'PC1'], data_reduced.loc[mislabeled_indices, 'PC2'], c='orchid')

plt.scatter(data_reduced.loc[mislabeled_legendary_indices, 'PC1'], data_reduced.loc[mislabeled_legendary_indices, 'PC2'], c='mediumspringgreen')



plt.xlabel("PC1")

plt.ylabel("PC2")

plt.legend(['Non-Legendary', 'Legendary', 'Non-Legendary Misclassified', 'Legendary Misclassified'])

plt.title("PCA Scatter Plot")

plt.show()