import pandas as pd

import tensorflow as tf

print(tf.__version__)

print(pd.__version__)
# load data

data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"

data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"



df_train = pd.read_csv(data_train_file)

df_test = pd.read_csv(data_test_file)



print(df_train.size)

print(df_test.size)



df_train.head()
# in first columns are labels, others are pixels

x_train_feature = df_train[df_train.columns[1:]]

y_train_feature_label = df_train[df_train.columns[0]]



x_test_feature = df_test[df_test.columns[1:]]

y_test_test_label = df_test[df_test.columns[0]]
x_train_feature.head()
y_train_feature_label.head()
#normalization of train data

x_train_feature = x_train_feature / 255.0

# load and normalization of test data



x_test_feature = df_test[df_test.columns[1:]]

y_test_feature_label = df_test[df_test.columns[0]]



x_test_feature = x_test_feature / 255.0

# define sequential model

model = tf.keras.models.Sequential()



# parametrize first (input) layer (128 units/neurons , )

model.add(tf.keras.layers.Dense(units=128, activation='relu', input_shape=(784, )))



# adding dropout layer (randomly setting neuron in layer to zero). To lower chance of overfitting

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Dense(units=128, activation='relu'))

model.add(tf.keras.layers.Dropout(0.5))



model.add(tf.keras.layers.Dense(units=256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.4))







model.add(tf.keras.layers.Dense(units=256, activation='relu'))

model.add(tf.keras.layers.Dropout(0.3))



model.add(tf.keras.layers.Dense(units=128, activation='relu'))



#output layer :  units  - the same as clases/labels

model.add(tf.keras.layers.Dense(units=10, activation='softmax'))



# compile ; optimizer Adam ; loss: Sparse Softmax (categorical) crossentrophy

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
# training the model 

model.fit(x_train_feature, y_train_feature_label, epochs=60)



# model evaluation

test_loss, test_accuracy = model.evaluate(x_test_feature, y_test_feature_label)