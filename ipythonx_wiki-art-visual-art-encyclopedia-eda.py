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

        plt.title(os.path.basename(os.path.normpath(path)))

        plt.xticks([])

        plt.yticks([])

    

    plt.show()
BASE_PATH = '../input/wikiart-gangogh-creating-art-gan/'

abstract_paint        = os.path.join(BASE_PATH, 'abstract')

animal_paint          = os.path.join(BASE_PATH, 'animal-painting')

cityspace_paint       = os.path.join(BASE_PATH, 'cityscape')

figurative_paint      = os.path.join(BASE_PATH, 'figurative')

flower_paint_paint    = os.path.join(BASE_PATH, 'flower-painting')

genre_painting_paint  = os.path.join(BASE_PATH, 'genre-painting')

landscape_paint       = os.path.join(BASE_PATH, 'landscape')

marina_paint          = os.path.join(BASE_PATH, 'marina')

myth_paint_paint      = os.path.join(BASE_PATH, 'mythological-painting')

nude_paint_paint      = os.path.join(BASE_PATH, 'nude-painting-nu')

portrait_paint        = os.path.join(BASE_PATH, 'portrait')

religious_paint_paint = os.path.join(BASE_PATH, 'religious-painting')

still_life_paint      = os.path.join(BASE_PATH, 'still-life')

symbolic_paint        = os.path.join(BASE_PATH, 'symbolic-painting')



print(f'Abstract images      : {len(os.listdir(abstract_paint))}')

print(f'Animal images        : {len(os.listdir(animal_paint))}')

print(f'Cityspace images     : {len(os.listdir(cityspace_paint))}')

print(f'Figurative images    : {len(os.listdir(figurative_paint))}')

print(f'Flower images        : {len(os.listdir(flower_paint_paint))}')

print(f'Genre images         : {len(os.listdir(genre_painting_paint))}')

print(f'Landscape images     : {len(os.listdir(landscape_paint))}')

print(f'Marina images        : {len(os.listdir(marina_paint))}')

print(f'Mythological images  : {len(os.listdir(myth_paint_paint))}')

print(f'Nude images          : {len(os.listdir(nude_paint_paint))}')

print(f'Portrait images      : {len(os.listdir(portrait_paint))}')

print(f'Religious images     : {len(os.listdir(religious_paint_paint))}')

print(f'Still Life images    : {len(os.listdir(still_life_paint))}')

print(f'Symbolic images      : {len(os.listdir(symbolic_paint))}')
vis(abstract_paint, 12, is_random=True)
vis(flower_paint_paint, 12, is_random=True)
vis(landscape_paint, 12, is_random=True)
vis(still_life_paint, 12, is_random=True)
vis(figurative_paint, 12, is_random=True)
def getImageMetaData(file_path):

    with Image.open(file_path) as img:

        img_hash = imagehash.phash(img)

        return img.size, img.mode, str(img_hash), file_path
from joblib import Parallel, delayed

from PIL import Image

import imagehash, glob, psutil, zipfile, cv2, math, random, warnings, os



img_meta_land = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(

    (delayed(getImageMetaData)(fp) for fp in glob.glob('../input/wikiart-gangogh-creating-art-gan/landscape/*.jpg'))

)

img_meta_land = pd.DataFrame(np.array(img_meta_land))

img_meta_land.columns = ['Size', 'Mode', 'Hash', 'Path']
print(img_meta_land.shape)

img_meta_land.head()