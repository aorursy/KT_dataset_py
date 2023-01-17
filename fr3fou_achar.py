import numpy as np # linear algebra
import random
import os
import tensorflow.keras as keras

base_metadata_path='/kaggle/input/tagged-anime-illustrations/danbooru-metadata/danbooru-metadata'
base_image_path='/kaggle/input/tagged-anime-illustrations/danbooru-images/danbooru-images'
def fix_dim(img):
    if len(img.shape) is 3:
        return img
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = img
    ret[:, :, 1] = img
    ret[:, :, 2] = img
    return ret
import keras.preprocessing.image as kimg 

def load_image(path):
    x = kimg.load_image(path)
    x = kimg.image_to_array(x)
    return fix_dim(x)
mods = os.listdir(base_image_path)

fig = plt.figure(figsize=(10,10))
for i in range(9):
    mod = os.path.join(base_image_path, random.choice(mods))
    filename = os.path.join(mod, random.choice(os.listdir(mod)))
    image = mpimg.imread(filename)
    image = fix_dim(image)
    fig.add_subplot(3, 3, i + 1)
    plt.imshow(image)
plt.show()
class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=32, dim=(512, 512), n_channels=3, n_classes=10, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return 

    def __getitem__(self, index):
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
import json
import time

def image_gen(batch_size=64):
    x = []
    y = []
    
    batcher = get_one()
    
    while True:
        x, y = next(batcher)
        
        x.append(x)
        y.append(y)
        
        if len(x) == batch_size:
            yield x, y
            x = []
            y = []
        

def get_one():   
    for file in os.listdir(base_metadata_path):
        with open(os.path.join(base_metadata_path, file), 'r') as f:
            for i, line in enumerate(f):
                j = json.loads(line)
                
                # get json fields
                image_id = j['id']
                ext = j['file_ext']
                tags = j['tags']
            
                # get tag names and ids
                tag_names = list(map(lambda t: t['name'], tags))
            
                # dir of the image
                image_path = str(int(image_id) % 1000).zfill(4)
            
                # path to image
                path = os.path.join(base_image_path, image_path, image_id) + f'.{ext}'
                # due to the smaller subset, not all images are available (?)
                if os.path.exists(path):
                    x = load_image(path)
                    y = tag_names
                    yield x, y