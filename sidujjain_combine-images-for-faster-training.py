import numpy as np 
import pandas as pd 
from pathlib import Path
import os
import PIL
import matplotlib.pyplot as plt
from tqdm import tqdm

PATH = Path("../input")
TRAIN = PATH/'train'
TEST = PATH/'test'
train_names = list({f[:36] for f in os.listdir(TRAIN)})
test_names = list({f[:36] for f in os.listdir(TEST)})

print(len(train_names), len(test_names))
CHANNELS = np.array(['green', 'red', 'blue', 'yellow'])
CHANNEL_CMAP = {"green": "Greens", "red": "Reds", "blue": "Blues", "yellow": "Oranges"}
def load_image(img_id, channels, img_dir, suffix='.png', size=512):
    px = np.zeros(shape=(len(channels),size,size))
    for i, ch in enumerate(channels):
        fname = str(img_dir/f'{img_id}_{ch}{suffix}')
        im = PIL.Image.open(fname)
        if size < 512:
            im = im.resize((size, size))
        px[i,:,:] = np.array(im)
    px = np.moveaxis(px.astype(np.uint8), 0, 2)
    return PIL.Image.fromarray(px)

def show_image(img, channels, title="", subax=None, figsize=(16,5)):
    px = np.array(img) / 255.
    px = np.moveaxis(px, 2, 0)
    if subax==None: fig, subax = plt.subplots(1, len(channels), figsize=figsize)
    for i, ch in enumerate(channels):
        subax[i].imshow(px[i], cmap=CHANNEL_CMAP[ch])
        if i == 0: subax[i].set_title(str(title))

def save_img(img_id, img, ch, path, suffix=".png", save=True):
    fname = str(path/f'{img_id}_{ch}{suffix}')
    if save:
        img.save(fname)
    return fname

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
img_id = train_names[0]
channels = CHANNELS
img = load_image(img_id, CHANNELS, TRAIN, size=299)
show_image(img, channels)
make_dir("tmp")
channels = CHANNELS
save_channel = "".join([ch[0] for ch in channels]);

fname = f'tmp/{img_id}_{save_channel}.png'
print("Saving to", fname)

img.save(fname)
img_read = PIL.Image.open(fname)
print("Same images", np.allclose(img, img_read))
print("image shape", np.array(img_read).shape)
channels = CHANNELS[[0, 1, 3]]
save_channel = "".join([ch[0] for ch in channels]);
make_dir(f"test_{save_channel}")
# Kaggle doesn't seem to create folder, but output files are visible
# Running for just 10 test imags
for img_id in tqdm(test_names[0:5]):
    img = load_image(img_id, channels, TEST, size=512)
    fname = f'test_{save_channel}/{img_id}_{save_channel}.png'
    img.save(fname)
save_channel = "".join([ch[0] for ch in channels]);
make_dir(f"train_{save_channel}")
# Kaggle doesn't seem to create folder, but output files are visible
# Running for just 10 train imags
for img_id in tqdm(train_names[0:5]):
    img = load_image(img_id, channels, TRAIN, size=512)
    fname = f'train_{save_channel}/{img_id}_{save_channel}.png'
    img.save(fname)
