import pandas as pd
import os

os.listdir('../input')
df = pd.read_csv('../input/data.csv')
df.head()
rdf = df[['age_month', 'spent_seconds', 'is_from_zp', 'is_studied_in_zp', 'is_studied_on_courses', 'edu_scores_100', 'lang_php',

   'lang_python', 'lang_js', 'is_graduated', 'is_graduation_course', 'approved']]
dataset = rdf.values
from sklearn import preprocessing
X = dataset[:, 0:dataset.shape[1] - 1]

Y = dataset[:, dataset.shape[1] - 1]
Y
min_max_scaler = preprocessing.MinMaxScaler()

X_scale = min_max_scaler.fit_transform(X)
X_scale
from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val_and_test
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense

model = Sequential([

    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid'),

])

from keras.optimizers import SGD

opt = SGD()

model.compile(optimizer='sgd',

              loss='binary_crossentropy',

              metrics=['accuracy'])
hist = model.fit(X_train, Y_train,

          batch_size=16, epochs=100,

          validation_data=(X_val, Y_val))
model.evaluate(X_test, Y_test)[1]
model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)
model.save_weights("model.h5")