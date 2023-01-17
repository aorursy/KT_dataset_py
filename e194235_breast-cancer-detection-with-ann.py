import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head()
df.info()
df = df.drop(["Unnamed: 32", "id"], axis=1)
df.describe().transpose()
sns.countplot(x="diagnosis", data=df)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])
le.classes_
f, ax = plt.subplots(figsize=(16, 12))

plt.title('Pearson Correlation Matrix', fontsize=25)



sns.heatmap(df.corr(), linewidths=0.25, vmax=0.7, square=True, cmap="BuGn",

            linecolor='w', annot=True, annot_kws={"size":8}, cbar_kws={"shrink": .9});
#drop_features = ["radius_mean", "perimeter_mean", "perimeter_se", "area_se", "radius_worst", "area_worst"]

drop_features = ["radius_mean", "perimeter_mean", "perimeter_se", "radius_se", "radius_worst", "perimeter_worst"]

#drop_features = ["area_mean", "perimeter_mean", "area_se", "perimeter_se", "perimeter_worst", "area_worst"]

df = df.drop(drop_features, axis=1)
df.head()
df.corr()["diagnosis"].sort_values().plot(kind="bar")
X = df.drop("diagnosis", axis=1).values

y = df["diagnosis"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)
X_train.shape
X_val.shape
X_test.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation, Dropout
model = Sequential()



model.add(Dense(24, activation="relu"))

model.add(Dropout(0.2))



model.add(Dense(16, activation="relu"))

model.add(Dropout(0.5))



model.add(Dense(1, activation="sigmoid"))



model.compile(loss="binary_crossentropy", optimizer="adam")
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
check_point = ModelCheckpoint("best_model.h5", monitor="val_loss", verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=1000)
model.fit(x=X_train, y=y_train,

          epochs=10000, callbacks=[check_point, early_stop],

          validation_data=(X_val,y_val), verbose=1)
losses = pd.DataFrame(model.history.history)
losses.plot()
losses["val_loss"].min()
from keras.models import load_model
saved_model = load_model('best_model.h5')
predictions = saved_model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))