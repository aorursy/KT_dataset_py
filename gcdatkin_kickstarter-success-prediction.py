import numpy as np

import pandas as pd



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight



import tensorflow as tf
data = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv')
data
data.info()
unneeded_columns = ['ID', 'name']



data = data.drop(unneeded_columns, axis=1)
data.isna().sum()
data['usd pledged'] = data['usd pledged'].fillna(data['usd pledged'].mean())
data.isna().sum().sum()
data['state'].unique()
data = data.drop(data.query("state != 'failed' and state != 'successful'").index, axis=0).reset_index(drop=True)
data['state'].unique()
data
data['deadline_year'] = data['deadline'].apply(lambda x: np.float(x[0:4]))

data['deadline_month'] = data['deadline'].apply(lambda x: np.float(x[5:7]))



data['launched_year'] = data['launched'].apply(lambda x: np.float(x[0:4]))

data['launched_month'] = data['launched'].apply(lambda x: np.float(x[5:7]))



data = data.drop(['deadline', 'launched'], axis=1)
data['state'] = data['state'].apply(lambda x: 1 if x == 'successful' else 0)
{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['category', 'main_category', 'currency', 'country'],

    ['cat', 'main_cat', 'curr', 'country']

)
data
y = data.loc[:, 'state']

X = data.drop('state', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=34)
X.shape
y.mean()
class_weights = class_weight.compute_class_weight(

    'balanced',

    y_train.unique(),

    y_train

)



class_weights = dict(enumerate(class_weights))

class_weights
inputs = tf.keras.Input(shape=(221,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=[

        'accuracy',

        tf.keras.metrics.AUC(name='auc')

    ]

)





batch_size = 64

epochs = 100



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    class_weight=class_weights,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=3,

            restore_best_weights=True,

            verbose=1

        )

    ],

    verbose=2

)
model.evaluate(X_test, y_test)