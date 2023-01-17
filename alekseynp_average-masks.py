import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
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

df['Label'] = df['Image_Label'].apply(lambda x: x.split('_')[1])

labels = sorted(list(df['Label'].unique()))
for l in labels:

    m = np.zeros((1400, 2100), dtype=np.float)

    count = 0



    for idx, row in df.iterrows():

        if row['Label'] == l:

            if not isinstance(row['EncodedPixels'], float):

                rle = row['EncodedPixels']

                mask = rle_to_mask(rle, 2100, 1400)

                mask = np.clip(mask, 0, 1)

                m += mask.astype(np.float)

            count += 1



    view = m / count



    plt.figure()

    plt.imshow(view)

    plt.title(l)

    

    plt.figure()

    plt.hist(view.flatten(), bins=100)

    plt.title('Histogram of Values for {}'.format(l))