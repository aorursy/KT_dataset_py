import numpy as np

import pandas as pd



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



from sklearn.neural_network import MLPClassifier

import tensorflow as tf
data = pd.read_csv("../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv")
data
data.drop('Date', axis=1, inplace=True)
data.isnull().sum()
data.drop(['Own goals', 'Own goal Time'], axis=1, inplace=True)
data['1st Goal'] = data['1st Goal'].fillna(data['1st Goal'].mean())
data.dtypes
print(f"Team: {data['Team'].unique()}\n")

print(f"Opponent: {data['Opponent'].unique()}\n")

print(f"Man of the Match: {data['Man of the Match'].unique()}\n")

print(f"Round: {data['Round'].unique()}\n")

print(f"PSO: {data['PSO'].unique()}\n")
label_encoder = LabelEncoder()



data['Man of the Match'] = label_encoder.fit_transform(data['Man of the Match'])

man_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}



data['PSO'] = label_encoder.fit_transform(data['PSO'])

pso_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}
data
round_values = list(data['Round'].unique())

round_values
round_mappings = {label: index for index, label in enumerate(round_values)}

round_mappings
data['Round'] = data['Round'].apply(lambda x: round_mappings[x])
data['Team'].unique()
data['Opponent'].unique()
pd.get_dummies(data['Team'])
pd.get_dummies(data['Opponent'].apply(lambda x: "opp_" + x))
data['Opponent'] = data['Opponent'].apply(lambda x: "opp_" + x)
data
data_concat = pd.concat([data, pd.get_dummies(data['Team']), pd.get_dummies(data['Opponent'])], axis=1)
data_concat.drop(['Team', 'Opponent'], axis=1, inplace=True)
data_concat
np.sum(data_concat.dtypes == 'object')
y = data_concat['Man of the Match']

X = data_concat.drop('Man of the Match', axis=1)
scaler = RobustScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
sk_model = MLPClassifier(hidden_layer_sizes=(16, 16))

sk_model.fit(X_train, y_train)
inputs = tf.keras.Input(shape=(85,))

x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(inputs)

x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(x)

outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(x)



tf_model = tf.keras.Model(inputs=inputs, outputs=outputs)
tf_model.compile(

    optimizer='adam',

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=['accuracy']

)
tf_model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=32,

    epochs=10

)
sk_score = sk_model.score(X_test, y_test)

tf_score = tf_model.evaluate(X_test, y_test, verbose=False)
print(f"   sklearn Model: {sk_score}")

print(f"TensorFlow Model: {tf_score[1]}")
X_test.shape