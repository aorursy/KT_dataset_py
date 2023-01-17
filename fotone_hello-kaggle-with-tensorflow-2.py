import pandas as pd 

import tensorflow as tf



def preprocessor(df, train_data):

    df = df.copy()

    df["Sex"] = (df["Sex"] == "male").astype(int)

    df["Age"] = df["Age"].fillna(method='pad')

    keys = ["Age", "Fare", "SibSp", "Parch", "Pclass"]



    for key in keys:

        item_mean = train_data[key].mean()

        item_std = train_data[key].std()

        df[key] -= item_mean

        df[key] /= item_std



    return df
original_train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



train = preprocessor(original_train, original_train)

test = preprocessor(test, original_train)



train_x = train[["Sex", "Fare", "SibSp", "Parch", "Pclass"]]

train_y = train["Survived"]



test_x = test[["Sex", "Fare", "SibSp", "Parch", "Pclass"]]
model = tf.keras.Sequential()



model.add(tf.keras.layers.Dense(units=52, activation='relu', input_shape=(5,)))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=39, activation='relu'))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=26, activation='relu'))

model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))



opti = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=opti, loss='binary_crossentropy', metrics=['acc'])



model.summary()
hist  = model.fit(train_x, train_y, batch_size=100, epochs=200, verbose=0)

import matplotlib.pyplot as plt

plt.plot(hist.history['loss'], 'b-', label='loss')

plt.show()

plt.plot(hist.history['acc'], 'b-', label='loss')

plt.show()
result = model.predict(test_x)



result = [1 if value >= 0.5 else 0 for value in result]



submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

submission['Survived'] = result



submission.to_csv('submission.csv', index = False)