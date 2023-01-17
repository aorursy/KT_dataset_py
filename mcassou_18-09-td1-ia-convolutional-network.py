# NumPy : librairie de calcul scientifique
import numpy as np

# MatPlotLib : librairie de visualisation et graphiques
from matplotlib import pyplot as plt
# Importation des données CIFAR10
from keras.datasets import cifar10

(X_train, _), (X_test, _) = cifar10.load_data()
# Normalisation entre 0 et 1
# - le deep learning a un peu de mal avec les grandes valeurs alors on veut normaliser nos valeurs entre 0 et 1
# - il se trouve que nous valeurs sont des couleurs (RGB) donc leur valeur maximale possible est 255
# - on a donc juste à diviser par 255 pour normaliser nos valeurs entre 0 et 1

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

print("x_train shape: ",X_train.shape)
print("x_test shape: ",X_test.shape)
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D
from keras.models import Model


input_img = Input(shape=(32, 32, 3)) # différent en fonction de la shape de nos images
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPool2D((2, 2), padding='same')(x)
# Nouvelle couche 2D avec 8 filtres :
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPool2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

encoded = MaxPool2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same', name='d1')(encoded)
x = UpSampling2D((2, 2), name='d2')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same', name='d3')(x)
x = UpSampling2D((2, 2), name='d4')(x)
# On n'enlève pas le padding à same ici car on veut du 32 par 32, et pas du 28 par 28
x = Conv2D(16, (3, 3), activation='relu', padding='same', name='d5')(x)
x = UpSampling2D((2, 2), name='d6')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='d7')(x) # decoded avec 3 filtre (1er paramètre) car on veut du 32,32,3 et non du 32,32,1

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()
# ...
# encoded : representation de notre vecteur qu'on veut recup
encoder = Model(input_img, encoded)
encoder.summary()
# Decoder :
# - cherche couche par leur nom
# - prend en entrée du 4,4,8 et sort 32,32,3

encoded_input = Input(shape=(4,4,8))
decoder_layer = autoencoder.get_layer('d1')(encoded_input)
decoder_layer = autoencoder.get_layer('d2')(decoder_layer)
decoder_layer = autoencoder.get_layer('d3')(decoder_layer)
decoder_layer = autoencoder.get_layer('d4')(decoder_layer)
decoder_layer = autoencoder.get_layer('d5')(decoder_layer)
decoder_layer = autoencoder.get_layer('d6')(decoder_layer)
decoder_layer = autoencoder.get_layer('d7')(decoder_layer)
decoder = Model(encoded_input, decoder_layer)
decoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Entraîne sur X époques (ici 300 époques)

autoencoder.fit(X_train, X_train,
                epochs=300,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test))

# Sur cb d'époques s'entraine t-on ? Tant que ta val_loss décroît mais attention à se qu'elle ne stagne pas pas ce qui engendrait du surentraînement
# On encode nos vecteurs :
encoded_imgs = encoder.predict(X_test)
# Prends en entrées nos vecteurs encodés pour sortir images décodées :
decoded_imgs = decoder.predict(encoded_imgs)
%matplotlib inline
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, ncols=10,figsize=(20, 4))
plt.gray()
for indice, row in enumerate(ax):
    for indice2, col in enumerate(row):
        if indice == 0:
            col.imshow(X_test[indice2].reshape(32,32,3)) # Ici on veut reshape en 32,32,3 (si on voulait 28,28,1 on ne serait pas obliger de mettre le 1)
        else:
            col.imshow(decoded_imgs[indice2].reshape(32,32,3))

plt.show()