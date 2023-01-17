import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import math, os, random, cv2

import numpy as np, pandas as pd

import matplotlib.pyplot as plt
#took form:https://www.kaggle.com/ihelon/monet-eda

def vis(path, n_images, is_random=True, figsize=(16, 16)):

    plt.figure(figsize=figsize)

    

    w = int(n_images ** .5)

    h = math.ceil(n_images / w)

    

    image_names = os.listdir(path)

    

    for i in range(n_images):

        image_name = image_names[i]

        if is_random:

            image_name = random.choice(image_names)

            

        img = cv2.imread(os.path.join(path, image_name))

        plt.subplot(h, w, i + 1)

        plt.imshow(img)

        plt.xticks([])

        plt.yticks([])

    

    plt.show()
BASE_PATH = '../input/gan-getting-started/'

MONET_PATH = os.path.join(BASE_PATH, 'monet_jpg')

PHOTO_PATH = os.path.join(BASE_PATH, 'photo_jpg')



print(f'Monet images: {len(os.listdir(MONET_PATH))}')

print(f'Photo images: {len(os.listdir(PHOTO_PATH))}')
vis(MONET_PATH, 16, is_random=True)
vis(PHOTO_PATH, 16, is_random=True)
BASE_PATH = '../input/van-gogh-paintings/'

Arles_PATH    = os.path.join(BASE_PATH, 'Arles')

Auvers_PATH   = os.path.join(BASE_PATH, 'Auvers sur Oise')

Drawings_PATH = os.path.join(BASE_PATH, 'Drawings')

Face_PATH     = os.path.join(BASE_PATH, 'Face')

Nuenem_PATH   = os.path.join(BASE_PATH, 'Nuenen')

Paris_PATH    = os.path.join(BASE_PATH, 'Paris')

Saint_PATH    = os.path.join(BASE_PATH, 'Saint Remy')

Sketches_PATH = os.path.join(BASE_PATH, 'Sketches in letters')

Villege_PATH  = os.path.join(BASE_PATH, 'Villege')

Water_PATH    = os.path.join(BASE_PATH, 'Watercolors')

YoungVan_PATH = os.path.join(BASE_PATH, 'Works of the young van Gogh')





print(f'Arles images   : {len(os.listdir(Arles_PATH))}')

print(f'Auvers images  : {len(os.listdir(Auvers_PATH))}')

print(f'Drawings images: {len(os.listdir(Drawings_PATH))}')

print(f'Face images    : {len(os.listdir(Face_PATH))}')

print(f'Nuenem images  : {len(os.listdir(Nuenem_PATH))}')

print(f'Paris images   : {len(os.listdir(Paris_PATH))}')

print(f'Saint images   : {len(os.listdir(Saint_PATH))}')

print(f'Sketches images: {len(os.listdir(Sketches_PATH))}')

print(f'Villege images : {len(os.listdir(Villege_PATH))}')

print(f'Water images   : {len(os.listdir(Water_PATH))}')

print(f'YoungVan images: {len(os.listdir(YoungVan_PATH))}')
vis(Arles_PATH, 16, is_random=True)
vis(Auvers_PATH, 16, is_random=True)
vis(Drawings_PATH, 16, is_random=True)
vis(Nuenem_PATH, 16, is_random=True)
vis(Paris_PATH, 16, is_random=True)
vis(Saint_PATH, 16, is_random=True)
vis(Sketches_PATH, 16, is_random=True)
vis(Villege_PATH, 16, is_random=True)
vis(Water_PATH, 16, is_random=True)
vis(YoungVan_PATH, 16, is_random=True)