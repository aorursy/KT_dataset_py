from tensorflow.keras.datasets.fashion_mnist import load_data

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt
(X_train_full, y_train_full), (X_test, y_test) = load_data()

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_train_full, X_test = X_train_full / 255, X_test / 255

X_valid, y_valid = X_train_full[:5000], y_train_full[:5000]

X_train, y_train = X_train_full[5000:],y_train_full[5000:]
# Random Forest instanziieren

rf_model = RandomForestClassifier(n_estimators=64, n_jobs=-1)
# Aus unseren 2D-Bildern (mit 28x28 Pixeln) machen wir gleich einen flachen

# Input Vektor mit 784 Werten (=28*28). Die erste Dimension lassen wir einfach

# so, wie sie bisher war (-1)

X_train_flat = X_train.reshape(-1, 28*28)
# Random Forest mit Hilfe unserer Trainingsdaten anpassen

rf_model.fit(X_train_flat, y_train)
# Test-Daten mit dem neuen Modell klassifizieren

y_pred = rf_model.predict(X_test.reshape((-1, 28*28)))
# Vergleichen der Ergebnisse mit den tatsächlichen Test-Labels - schaut gut

# aus. Bei den ersten 10 Datensätzen stimmen alle Werte überein

y_pred[0:10], y_test[0:10]
# Evaluieren der Modellqualität auf Basis der Accuracy

print("Accuracy Score:", accuracy_score(y_test, y_pred))
# Platz für deinen Code