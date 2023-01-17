from tensorflow.keras.datasets.fashion_mnist import load_data

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten

from tensorflow.keras.losses import sparse_categorical_crossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import sparse_categorical_accuracy, sparse_categorical_crossentropy

import tensorflow as tf

from matplotlib import pyplot as plt
(X_train_full, y_train_full), (X_test, y_test) = load_data()

labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

X_train_full, X_test = X_train_full / 255, X_test / 255

X_valid, y_valid = X_train_full[:5000], y_train_full[:5000]

X_train, y_train = X_train_full[5000:],y_train_full[5000:]
# Neues, künstliches neuronales Netz anlegen

model = Sequential()
# Einfügen eines Input Layers, der unsere 28x28 Bilder übernehmen kann.

# Flatten ist ein Layer, der matrixförmige Inputs in einen Vektor umwandelt

model.add(Flatten(input_shape=[28, 28]))
# Einfügen von zwei Hidden Layern, die unserem Netz genügend Freiheitsgrade

# geben, um Klassifikation zu ermöglichen. Dense ist ein vollvernetzer ANN

# Layer. Relu ist eine einfach, aber effektive Aktivierungsfunktion

model.add(Dense(300, activation="relu"))

model.add(Dense(100, activation="relu"))
# Einfügen eines Output Layers, der die Netzaktivierung auf unsere 10 Klassen

# reduziert. Softmax ist eine AKtivierungsfunktion, die den Input über eine

# Wahrscheinlichkeitsfunktion normalisiert (soft an den Enden, wahrscheinlich

# im Zentrum bzw. beim Erwartungswert)

model.add(Dense(10, activation="softmax"))
# Finales Netz betrachten

model.summary()
# Netz kompilieren. Dabei legen wir drei wichtige Parameter fest. Diese sind

# der Optimizer, der verwendet wird, um die optimale Netzperformanz zu finden,

# die Metrik, die zur Ermittlung des Trainings/Test-Loss verwendet wird und

# die Metriken, die während des Trainings aufgezeichnet werden sollen

model.compile(

    loss=sparse_categorical_crossentropy,

    optimizer=Adam(),

    metrics=[

        sparse_categorical_accuracy,

        sparse_categorical_crossentropy

    ]

)
# Größe unserer Trainings-Batches festlegen

batch_size = 64
# Vorbereiten der Trainings/Validation/Test-Daten als Batches

test = (X_test,  y_test)

valid = (X_valid, y_valid)

train = (X_train, y_train)



train_ds = tf.data.Dataset.from_tensor_slices(train).batch(batch_size)

valid_ds = tf.data.Dataset.from_tensor_slices(valid).batch(batch_size)

test_ds  = tf.data.Dataset.from_tensor_slices(test).batch(batch_size)
# Netz trainieren

train_history = model.fit(

    train_ds,

    validation_data=valid_ds,

    epochs=10,

    verbose=2

)
# Testen des Modells mit neuen Daten

test_history = model.evaluate(test_ds)
# Betrachten wir das Lernverhalten unserers Netzes noch etwas genauer

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))



ax[0].plot(train_history.history["loss"], label="train")

ax[0].plot(train_history.history["val_loss"], label="validation")

ax[0].hlines(

    y=test_history[0],

    xmin=0,

    xmax=len(train_history.history["loss"]),

    colors="green",

    label="test"

)

ax[0].legend(loc=0)

ax[0].set_title("Loss")



ax[1].plot(train_history.history["sparse_categorical_accuracy"], label="train")

ax[1].plot(train_history.history["val_sparse_categorical_accuracy"], label="validation")

ax[1].hlines(

    y=test_history[1],

    xmin=0,

    xmax=len(train_history.history["loss"]),

    colors="green",

    label="test"

)

ax[1].legend(loc=0)

ax[1].set_title("Accuracy")



plt.show()
# Platz für deinen Code