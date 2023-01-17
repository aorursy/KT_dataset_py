import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
d = load_breast_cancer()

cancer_data = pd.DataFrame(load_breast_cancer()["data"], columns=d["feature_names"])
cancer_data["target"] = d["target"]
cancer_data
sns.countplot(x = "target", data=cancer_data)
cancer_data.info()
cancer_data.describe().transpose()
cancer_data.corr()["target"].sort_values()
cancer_data.corr()["target"][:-1].sort_values().plot(kind = "bar", figsize = (16,6))
plt.figure(figsize=(16,6))
sns.heatmap(cancer_data.corr(), cmap="rainbow", annot=True)
X = cancer_data.drop("target", axis=1).values
y = cancer_data["target"].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)
X_train.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(units = 30, activation = "relu"))
model.add(Dense(units = 15, activation = "relu"))
model.add(Dense(units = 1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy")
model.fit(X_train,y_train, epochs=300, validation_data=(X_test,y_test))
pd.DataFrame(model.history.history).plot(figsize =(16,6))
model = Sequential()
model.add(Dense(units = 30, activation = "relu"))
model.add(Dense(units = 15, activation = "relu"))
model.add(Dense(units = 1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy")
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(mode="min", verbose=1, patience=25)
model.fit(X_train, y_train, epochs=500, validation_data=(X_test,y_test), callbacks=[early_stop])
pd.DataFrame(model.history.history)
pd.DataFrame(model.history.history).plot(figsize = (16,6))
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(units = 30, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 15, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = 1, activation = "sigmoid"))
model.compile(optimizer = "adam", loss = "binary_crossentropy")
model.fit(X_train, y_train, epochs=500, validation_data=(X_test,y_test), callbacks=[early_stop])
pd.DataFrame(model.history.history)
pd.DataFrame(model.history.history).plot(figsize = (16,6))
predictions = model.predict_classes(X_test)
predictions
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
new_cancer_call = cancer_data.drop("target", axis = 1).iloc[568]
new_cancer_call.values.reshape(-1,30)
single_call = scalar.transform(new_cancer_call.values.reshape(-1,30) )
model.predict(single_call)
