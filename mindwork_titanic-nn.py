# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



work_path = '/kaggle/input/titanic/'



for dirname, _, filenames in os.walk(work_path):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(work_path + "train.csv")
test_data = pd.read_csv(work_path + "test.csv")
# Clearing/Mutating training data

# Remove those who have not the age, lol

train_data = train_data[train_data["Age"] > 0]

train_data = train_data[train_data["Embarked"].isin(['S', 'C', 'Q'])]



# train_data["Sex"] = train_data["Sex"].map({'male': 1, 'female': 0})

# test_data["Sex"] = test_data["Sex"].map({'male': 1, 'female': 0})

# test_data["Fare"] = test_data["Fare"].map(lambda fare: 0 if np.isnan(fare) else fare)

# test_data["Age"] = test_data["Age"].map(lambda age: 0 if np.isnan(age) else age)



# train_data = train_data.drop("Cabin", axis=1)

# test_data = test_data.drop("Cabin", axis=1)



# todo is it useful?

# pd.to_numeric(train_data["Fare"], downcast='float')

# pd.to_numeric(test_data["Fare"], downcast='float')




columns = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

label = "Survived"



train_label = train_data[label]





# Draw some graphs, examples from here https://machinelearningmastery.com/visualize-machine-learning-data-python-pandas/

# Scatter matrix

# pd_plt.scatter_matrix(train_data[columns])

# plt.show()



# Density

# train_data.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)

# plt.show()
def format_predictions(ndarray: np.ndarray):

    ndarray = np.where(np.isnan(ndarray), 0, ndarray)

    return np.where(ndarray > 0.5, 1, 0)





import tensorflow as tf

from tensorflow import feature_column



# tf.random.set_seed(42)

conf = tf.device('device:cpu:0')



feature_columns = [feature_column.numeric_column('Fare')]



# numeric cols



# bucketized cols

# todo think about learning tree for categorizing

age_buckets = feature_column.bucketized_column(feature_column.numeric_column('Age'), boundaries=[10, 20, 30, 40, 50, 60, 100])

feature_columns.append(age_buckets)



# categorical cols

sex = feature_column.categorical_column_with_vocabulary_list('Sex', train_data['Sex'].unique())

sex_one_hot = feature_column.indicator_column(sex)

feature_columns.append(sex_one_hot)



pclass = feature_column.categorical_column_with_vocabulary_list('Pclass', train_data['Pclass'].unique())

pclass_one_hot = feature_column.indicator_column(pclass)

feature_columns.append(pclass_one_hot)



embarked = feature_column.categorical_column_with_vocabulary_list('Embarked', train_data['Embarked'].unique())

embarked_one_hot = feature_column.indicator_column(embarked)

feature_columns.append(embarked_one_hot)



# categorical_column_with_identity() don't work properly, don't know why

name = feature_column.categorical_column_with_hash_bucket('Name', 10)

name_embedded = feature_column.embedding_column(name, 10)

feature_columns.append(name_embedded)



# Really tricky stuff, example from here https://towardsdatascience.com/end-to-end-machine-learning-with-tfx-on-tensorflow-2-x-6eda2fb5fe37

feature_inputs = {

    "Fare": tf.keras.layers.Input(name="Fare", shape=(), dtype=tf.float32),

    "Pclass": tf.keras.layers.Input(name="Pclass", shape=(), dtype=tf.int32),

    "Age": tf.keras.layers.Input(name="Age", shape=(), dtype=tf.int32),

    "Sex": tf.keras.layers.Input(name="Sex", shape=(), dtype=tf.string),

    "Embarked": tf.keras.layers.Input(name="Embarked", shape=(), dtype=tf.string),

    "Name": tf.keras.layers.Input(name="Name", shape=(), dtype=tf.string)

}



feature_layer = tf.keras.layers.DenseFeatures(feature_columns, trainable=False)(feature_inputs)

layer_1 = tf.keras.layers.BatchNormalization()(feature_layer)

layer_2 = tf.keras.layers.Dense(256, activation="relu")(layer_1)

layer_3 = tf.keras.layers.Dropout(0.2)(layer_2)

layer_4 = tf.keras.layers.Dense(128, activation="relu")(layer_3)

outputs = tf.keras.layers.Dense(1, activation="sigmoid")(layer_4)



model = tf.keras.Model(inputs=feature_inputs, outputs=outputs)



model.compile(optimizer=tf.keras.optimizers.SGD(),

              loss=tf.keras.losses.BinaryCrossentropy(),

              metrics=[tf.keras.metrics.Precision(),

                       tf.keras.metrics.FalsePositives(),

                       tf.keras.metrics.FalseNegatives()])



callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4)]



model.fit(x=dict(train_data), y=train_label, validation_split=0.3, epochs=2000, callbacks=callbacks)

# print(model.summary())



# sns.pairplot(train_data[features + [label]], diag_kind="kde")

# plt.show()

train_predictions = model.predict(x=dict(train_data))



# Calculate confusion matrix for predicted test values

print(tf.math.confusion_matrix(np.array(train_label), format_predictions(train_predictions)))



# For Kaggle submission predictions

# test_predictions = model.predict(x=dict(test_data))



# output = pd.DataFrame({'PassengerId': test_data["PassengerId"], 'Survived': format_predictions(test_predictions).flat})



# output.to_csv(work_path + 'neural_net.csv', index=False)

# print("Your submission was successfully saved!")