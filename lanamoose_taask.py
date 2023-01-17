import numpy as np

import pandas as pd

import keras



from keras.utils.np_utils import to_categorical

from sklearn.preprocessing import LabelEncoder

#pandas - для работы с данными, с csv

#sklearn - для преобразования текста в числа и обратно

import os



os.system("ls ../input")
#кодировщик labels(class_1, class_2...)

label_encoder = LabelEncoder()
train_df= pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

submit_df = pd.read_csv("../input/sampleSubmission.csv")
#вывести название столбцов

print(train_df.columns)

print(test_df.columns)

print(submit_df.columns)
#разделить обучающий набор данных на data и target без колонки id

data = train_df.drop(['target', 'id'], axis=1)

target = train_df['target']

encoded_target = label_encoder.fit_transform(target)

one_hot_target = to_categorical(encoded_target)
#удалить колонку 'id' в  тестовом наборе данных

test_df = test_df.drop(['id'], axis=1)
#вывести итоговые размеры таблиц

print(data.shape)

print(one_hot_target.shape)
#разбить обучающий dataset для cross-validation (4/5 для train и 1/5 для test)

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(data.index, one_hot_target, test_size=0.2, random_state=0)
#вывести результат разбиения

print("train size: {0}, test shape: {1}".format(x_train.shape, x_test.shape))

print(data.iloc[x_train])
# создать небольшую модель нейронной сети

from keras.models import Sequential

from keras.layers import Dense



model = Sequential()

model.add(Dense(units=30, activation='relu', input_dim=data.shape[1]))

model.add(Dense(units=10, activation='relu'))

model.add(Dense(units=9, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# обучить модель с помощью train dataset

model.fit(data.iloc[x_train], y_train, epochs=50, batch_size=32)
scores = model.evaluate(data.iloc[x_test], y_test)

# точность: ~ 77-79% (для небольшой нейронной сети, 4/5 от train dataset и 50 epochs)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#преобразовать обратно в labels

predicted_test = model.predict(data.iloc[x_test], batch_size=32)

print(label_encoder.inverse_transform(list(map(np.argmax, predicted_test))))
# вывести вероятности

print(predicted_test)
#обучить модель на полном train dataset

model.fit(data, one_hot_target, epochs=50, batch_size=32)
# предсказать вероятности

predicted_targets = model.predict(test_df, batch_size=32)
# вывести вероятости

print(predicted_targets)
# проверить кэшированные labels в label_encoder

label_encoder.classes_
# добавить итоговые вероятности в таблицу

submit_df[label_encoder.classes_] = predicted_targets

#и вывести результирующую таблицу

submit_df.head()
# сохранить результат

submit_df.to_csv('prediction.csv', index = False)