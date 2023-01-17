import matplotlib.pyplot as plt

import matplotlib.patches

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F
def rle_to_mask(rle_string, width, height):

    '''

    convert RLE(run length encoding) string to numpy array



    Parameters:

    rle_string (str): string of rle encoded mask

    height (int): height of the mask

    width (int): width of the mask



    Returns:

    numpy.array: numpy array of the mask

    '''



    rows, cols = height, width



    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]

        rle_pairs = np.array(rle_numbers).reshape(-1, 2)

        img = np.zeros(rows * cols, dtype=np.uint8)

        for index, length in rle_pairs:

            index -= 1

            img[index:index + length] = 255

        img = img.reshape(cols, rows)

        img = img.T

        return img
df = pd.read_csv('../input/understanding_cloud_organization/train.csv')

df.set_index('Image_Label', inplace=True)
rle = df.loc['0011165.jpg_Flower', 'EncodedPixels']

holes = [(485, 1030), (654, 1053)]

mask = rle_to_mask(rle, 2100, 1400)

mask = np.clip(mask, 0, 1)



plt.figure()

plt.imshow(mask)

ax = plt.gca()

fig, axs = plt.subplots(1, len(holes))



for h_idx, h in enumerate(holes):

    rect = matplotlib.patches.Rectangle((h[1]-20, h[0]-20), 40, 40, linewidth=1, edgecolor='red', facecolor='none')

    ax.add_patch(rect)

    axs[h_idx].imshow(mask[(h[0]-20):(h[0]+20), (h[1]-20):(h[1]+20)])

    

plt.show()
rle = df.loc['0011165.jpg_Fish', 'EncodedPixels']

holes = [(485, 1030), (654, 1053), (804, 833)]

mask = rle_to_mask(rle, 2100, 1400)

mask = np.clip(mask, 0, 1)



plt.figure()

plt.imshow(mask)

ax = plt.gca()

fig, axs = plt.subplots(1, len(holes))



for h_idx, h in enumerate(holes):

    rect = matplotlib.patches.Rectangle((h[1]-20, h[0]-20), 40, 40, linewidth=1, edgecolor='red', facecolor='none')

    ax.add_patch(rect)

    axs[h_idx].imshow(mask[(h[0]-20):(h[0]+20), (h[1]-20):(h[1]+20)])

    

plt.show()
kernel = torch.FloatTensor([

        [

            [1, 1, 1, 1, 1],

            [1, 1, 1, 1, 1],

            [1, 1,-8, 1, 1],

            [1, 1, 1, 1, 1],

            [1, 1, 1, 1, 1]

        ]

    ]).unsqueeze(1)
for i in range(100):

    row = df.iloc[i]

    rle = row['EncodedPixels']

    if not isinstance(rle, float):

        mask = rle_to_mask(rle, 2100, 1400)

        mask = np.clip(mask, 0, 1)

        out = F.conv2d(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float(), weight=kernel, padding=2, stride=1)

        holes = list(zip(*np.where(out[0, 0].numpy() == 24.)))

        if len(holes) > 0:

            print(row.name, holes)