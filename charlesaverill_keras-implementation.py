import numpy as np

import pandas as pd

import tensorflow as tf



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD, Adam



np.random.seed(50)
train = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_train.csv.zip")

test = pd.read_csv("../input/flight-delays-fall-2018/flight_delays_test.csv.zip")



train.head()
train["Month"] = [int(month[2:]) for month in train["Month"]]

train["DayofMonth"] = [int(day[2:]) for day in train["DayofMonth"]]

train['hour'] = train['DepTime'] // 100



train = train.drop("DepTime", axis=1)

train = train.drop("DayOfWeek", axis=1)



train["dep_delayed_15min"] = [1  if n == "Y" else 0 for n in train["dep_delayed_15min"]]



for col in ['Origin', 'Dest', 'UniqueCarrier']:

    train[col] = pd.factorize(train[col])[0]



print(test)



test["Month"] = [int(month[2:]) for month in test["Month"]]

test["DayofMonth"] = [int(day[2:]) for day in test["DayofMonth"]]

test['hour'] = test['DepTime'] // 100



test = test.drop("DepTime", axis=1)

test = test.drop("DayOfWeek", axis=1)



for col in ['Origin', 'Dest', 'UniqueCarrier']:

    test[col] = pd.factorize(test[col])[0]
features = train.drop(["dep_delayed_15min"], axis=1)

labels = train["dep_delayed_15min"]

features, labels = features.to_numpy(), labels.to_numpy()
print(features.shape)
model = None

model = Sequential([

    Dense(32, input_shape=(features.shape[1],), activation='relu'),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')

])

model.compile(optimizer=Adam(lr=.01), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(features, labels, epochs=15, batch_size=128, validation_split=.2, shuffle=True)
import matplotlib.pyplot as plt



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
predictions = model.predict(test)

preds = [pred[0] for pred in predictions]
submission = pd.DataFrame({'id':range(100000),'dep_delayed_15min':preds})

submission.to_csv("output.csv" ,index=False)

submission