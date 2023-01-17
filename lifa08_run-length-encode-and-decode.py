import numpy as np
import pandas as pd
from skimage.data import imread
import matplotlib.pyplot as plt
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
pixels = np.array((0, 1, 1, 1, 1, 0, 0, 0, 1))

# Concatenating a zero at the start and end of the array is to
# make sure that the first changing is always from 0 to 1
pixels = np.concatenate([[0], pixels, [0]])
print('pixels:', pixels)

# the array except the first element
print('pixels[1:]:', pixels[1:])
# the array except the last element
print('pixels[:-1]:', pixels[:-1])

# runs include indices to wherever 0s change to 1s or 1s change to 0s
print('where condition:', pixels[1:] != pixels[:-1])
runs = np.where(pixels[1:] != pixels[:-1])
print('runs:', runs)

# the purpose of adding 1 here is to make sure that the indices point to
# the very first 1s or 0s of the 1s or 0s, this is needed because
# np.where gets the indices of elements before changing
runs = runs[0] + 1
print('runs = runs[0] + 1:', runs)

# runs[1::2] --> runs[start:stop:step], thus 2 here is the step
# thus runs[1::2] includes the indices of the changing from 1 to 0
print('runs[1::2]:', runs[1::2])

# runs[::2] includes the indices for the changing from 0 to 1
print('runs[::2]:', runs[::2])

# the length of 1s
print('runs[1::2]-runs[::2]:', runs[1::2] - runs[::2])

# replace runs[1::2] with the lengths of consecutive 1s
runs[1::2] -= runs[::2]

print('return:', ' '.join(str(x) for x in runs))
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
mask_rle = ' '.join(str(x) for x in runs)
s = mask_rle.split()
print('s:', s)

print('s[0:][::2]:', s[0:][::2])
assert(s[0:][::2] == s[::2])

print('s[1:][::2]:', s[1:][::2])
assert(s[1:][::2] == s[1::2])

starts = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
print('starts:', starts)

rle_decode(mask_rle, (1, 9))
def rle_test():
    for i in range(100):
        data = np.random.randint(0, 2, (100,100))
        data_rle_enc = rle_encode(data)
        data_rle_dec = rle_decode(data_rle_enc, data.shape)
        np.testing.assert_allclose(data, data_rle_dec)
rle_test()
masks = pd.read_csv('../input/train_ship_segmentations.csv')
num_masks = masks.shape[0]
print('number of training images', num_masks)
masks.head()
def display_img_and_masks(ImageId, ImgShape=(768, 768)):
    img = imread('../input/train/' + ImageId)
    img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros(ImgShape)

    for mask in img_masks:
        # Note that NaN should compare as not equal to itself
        if mask == mask:
            all_masks += rle_decode(mask, ImgShape).T

    fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(all_masks)
    axarr[2].imshow(img)
    axarr[2].imshow(all_masks, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    plt.show()
# image that has ships
ImageId = '000155de5.jpg'
display_img_and_masks(ImageId)
# image that has no ship
ImageId = '00003e153.jpg'
display_img_and_masks(ImageId)