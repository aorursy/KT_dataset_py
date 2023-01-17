import os

import numpy as np

from PIL import Image

from matplotlib import pyplot as plt
# Chemin de l'image

img1_path = '../input/foot.jpg'



# Lecture de l'image avec le module 'Image'

img1_object = Image.open(img1_path)



# On convertit l'objet image en tableau

img1 = np.array(img1_object)
# On lit les dimensions du tableau (= résolution de l'image et nombre de couleurs)

height, width, pixel_nb_color = img1.shape

print("La résolution de l'image est de", height, "par", width, 'soit', width*height, 'pixels.')



# Affichage de l'image

plt.imshow(img1)
# On affiche la valeur du pixel à l'origine

print(img1[0,0])



# On affiche la valeur d'un pixel au hasard

print(img1[100,10])
# On s'amuse à multiplier la longeur de l'image par 2

new_dimension = (width*2, height)

img2_object = img1_object.resize(new_dimension)



# On convertit l'objet image en tableau

img2 = np.array(img2_object)

plt.imshow(img2)
# On enregistre cette image dans le répertoire courant

img2_path = 'foot_2.jpg'

img2_object.save(img2_path)