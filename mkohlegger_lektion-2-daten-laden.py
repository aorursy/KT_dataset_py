from tensorflow.keras.datasets.fashion_mnist import load_data

from matplotlib import pyplot as plt
# Daten laden

(X_train_full, y_train_full), (X_test, y_test) = load_data()
# Bilddaten zwischen 0 und 1 normieren. Derzeit sind die Farbwerte jedes

# einzelnen Pixels als werte zwischen 0 und 255 gespeichert. Das muss natürlich

# nur für die Bilddaten (X) gemacht werden

X_train_full = X_train_full / 255

X_test = X_test / 255
# Trainingsdaten in Training/Validation Daten splitten. WIr verwenden als 

# validation Set einfach die ersten 5000 Bilder im Datenset

X_valid = X_train_full[:5000]

y_valid= y_train_full[:5000]



X_train = X_train_full[5000:]

y_train = y_train_full[5000:]
# Labels der Daten als Liste instanziieren

labels = [

    "T-shirt/top",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandal",

    "Shirt",

    "Sneaker",

    "Bag",

    "Ankle boot"

]
# Erster Blick in die Trainings-Daten

X_train.shape, y_train.shape
# Erster Blick in die Validation-Daten

X_valid.shape, y_valid.shape
# Erster Blick in die Test-Daten

X_test.shape, y_test.shape
# Erster Blick auf die geladenen Bilder. Hierzu plotten wir wahllos die ersten

# 5 Bilder des Trainingsdatensets. Ich habe bewusst die Achseninformation

# nicht ausgeblendet, damit wir auch die Dimensionen der Bilder sehen



fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(20,20))



for i in range(5):

    ax[i].set_title(labels[y_train[i]])

    ax[i].imshow(X_train[i], cmap='gray_r')

    

plt.show()
# Platz für deinen Code