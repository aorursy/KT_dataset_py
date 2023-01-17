import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.optimizers import SGD
from keras import losses
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

print(os.listdir("../input"))

# Pandas Dataframes mit Ausgabe der ersten fünf Zeilen durch .head()
train_df = pd.read_csv("../input/train.csv", delimiter=",")
train_df.columns = ['x', 'y', 'Kategorie']
train_df.head()
# Aufsplittung der Trainings- und Testdaten durch .values werden Dataframes zu Numpyarrays
features = train_df[["x", "y"]].values 
labels = train_df["Kategorie"].values
x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=0, test_size = 0.2)

# Print nur zur Überprüfung ob Datenset auch wirklich gesplittet wurde.
print(len(x_train))
print(x_train.shape)
print(len(x_test))
# Punkte bzw. Daten plotten zur Visualisierung
colors = {0:'red',1:'blue'}

plt.scatter(features[:,0],features[:,1],c=train_df["Kategorie"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()
# Tensorflow bzw. Keras ausprobieren
# Baseline Modell definieren
epoch = 12
model = Sequential()
model.add(Dense(20, input_dim=2, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, decay=0.0),
              metrics=['binary_accuracy'])

# Modell mit Scaler (normalisieren) trainieren, um durch z.B. 0,1234 zu 0 zu runden. Erzielt dadurch auch eine genauere Klassifizerung. 
# Weniger als 6 Epochs liefern eine schlechtere Acc.
scaler = StandardScaler()
history = model.fit(scaler.fit_transform(x_train), y_train, validation_split=0.1, epochs=epoch)

# Testdaten laden, features und labels trennen
pred_pd = pd.read_csv("../input/test.csv")
pred_pd.columns = ['x', 'y', 'Kategorie']
x_test = pred_pd[["x", "y"]].values 
y_test = pred_pd["Kategorie"].values

# Vorhersage der Kategorie vom Modell
y_pred = model.predict_classes(scaler.transform(x_test))

# zeigt Prediction-Ergebnisse und stellt diese den Testwerten gegenüber
print("Ergebnisse der Prediction zu den jeweiligen X,Y-Werten der Testdatei.")
print("")
for i in range(len(x_test)):
    print("%s: X=%s, Predicted=%s" % (i, x_test[i], y_pred[i]))

# Vorhersage speichern
prediction = pd.DataFrame()
id = []
for i in range(len(x_test)):
    id.append(i)
    i = i + 1

# Struktur der Ausgabe
prediction["Id (String)"] = id 
prediction["Category (String)"] = y_pred.astype(int)
prediction.to_csv("predict.csv", index=False)
# Print der relevanten Werte der letzten Epoche.
last_idx = epoch -1
print("Acc & Loss beim Durchlauf der Trainingsdaten:")
print("Accuracy/Genauigkeit der letzten Epoche: ", history.history['binary_accuracy'][last_idx])
print("Loss/Verlust der letzten Epoche: ", history.history['loss'][last_idx])
print("")
print("Acc & Loss beim Durchlauf der Validierungsdaten:")
print("Validierte Acc der letzten Epoche: ", history.history['val_binary_accuracy'][last_idx])
print("Validierte Loss der letzten Epoche: ", history.history['val_loss'][last_idx])
print("")


# Genauigkeit plotten
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Loss plotten
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Punkte bzw. Daten plotten zur Visualisierung der Prediction
colors = {0:'red',1:'blue'}
plt.title("Predicted Data")


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
area = (15 * np.random.rand(N))**2  # 0 to 15 point radii
plt.scatter(x_test[:,0],x_test[:,1],c=pred_pd["Kategorie"].apply(lambda x: colors[x]),s=area)
#plt.scatter(x_test[:,0],x_test[:,1],c=pred_pd["Kategorie"].apply(lambda x: colors[x]))
plt.xlabel("x")
plt.ylabel("y")
plt.show()