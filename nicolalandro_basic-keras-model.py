import tensorflow as tf

import pandas as pd

from sklearn.model_selection import train_test_split



# Load the data

df = pd.read_csv("../input/train.csv")

# test = pd.read_csv("../input/test.csv")



y = df["label"]

x = df.drop(labels = ["label"],axis = 1) 



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

x_train, x_test = x_train / 255.0, x_test / 255.0



x_train = x_train.values.reshape(-1,28,28,1)

x_test = x_test.values.reshape(-1,28,28,1)



model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),

  tf.keras.layers.Dense(512, activation=tf.nn.relu),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=50)

model.evaluate(x_test, y_test)
df_test = pd.read_csv("../input/test.csv")

prediction = model.predict(df_test.values.reshape(-1,28,28,1)).argmax(1)

my_submission = pd.DataFrame({'ImageId': df_test.index + 1, 'Label': prediction})

my_submission.to_csv('submission.csv', index=False)