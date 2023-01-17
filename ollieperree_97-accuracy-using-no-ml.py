import numpy as np

import pandas as pd

from PIL import Image, ImageFilter

import matplotlib.pyplot as plt

from pathlib import Path

import skimage.measure

from sklearn.metrics import accuracy_score

from tqdm.notebook import tqdm

Path.ls = lambda p: list(p.iterdir())
train = Path("../input/fingers/train")

test = Path("../input/fingers/test")
train.ls()

fist = np.array(Image.open(train/"d4b08243-4cd3-493a-8616-e83c5ea23b7a_0R.png"))
fist_mask = fist > 80
plt.imshow(fist_mask)
for i in range(5):

    new_fist_mask = fist_mask.copy()

    for x in range(128):

        for y in range(128):

            for dx in range(-1, 2):

                for dy in range(-1, 2):

                    x_ = x + dx

                    y_ = y + dy

                    if 0 <= x_ < 128 and 0 <= y_ < 128:

                        if fist_mask[y_][x_]:

                            new_fist_mask[y][x] = True

    fist_mask = new_fist_mask
plt.imshow(fist_mask)
fist_mask[80:, ] = True
plt.imshow(fist_mask)
im = np.array(Image.open(train.ls()[140]))



fingers = im * (1-fist_mask) > 85

fingers = skimage.measure.label(fingers)
plt.imshow(fingers)
print(fingers.max(), "fingers")
test_y = []

test_pred = []

    

for fn in tqdm(test.ls()):

    im = Image.open(fn)

    test_y.append(int(fn.name[-6:-5]))

    fingers = im * (1-fist_mask) > 85

    fingers = skimage.measure.label(fingers)

    pred = fingers.max()

    test_pred.append(pred)
accuracy_score(test_y, test_pred)