import pandas as pd

import numpy as np



train = pd.read_csv('../input/understanding_cloud_organization/train.csv')



def rle2mask(rle_str, shape):

    if isinstance(rle_str, float):

        return np.zeros(shape, dtype=np.uint8)

    mask = [0 for _ in range(shape[0] * shape[1])]

    rle = [int(c) for c in rle_str.split(' ')]

    for i0, i1 in zip(rle[::2], rle[1::2]):

        for idx in [i for i in range(i0, i0 + i1)]:

            mask[idx] = 1

    mask = np.array(mask, dtype=np.uint8).reshape(shape[0], shape[1], order='F')

    return mask



def mask2rle(pred):

    shape = pred.shape

    mask = np.zeros([shape[0], shape[1]], dtype=np.uint8)

    points = np.where(pred == 1)

    if len(points[0]) > 0:

        mask[points[0], points[1]] = 1

        mask = mask.reshape(-1, order='F')

        pixels = np.concatenate([[0], mask, [0]])

        rle = np.where(pixels[1:] != pixels[:-1])[0]

        rle[1::2] -= rle[::2]

    else:

        return ''

    return ' '.join(str(r) for r in rle)



# Test RLE functions

assert mask2rle(rle2mask(train['EncodedPixels'].iloc[0], (1400, 2100))) == train['EncodedPixels'].iloc[0]

assert mask2rle(rle2mask('1 1', (1400, 2100))) == '1 1'

print('ALL DONE.')