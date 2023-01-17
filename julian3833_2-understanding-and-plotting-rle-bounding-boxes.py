import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../input/train_ship_segmentations_v2.csv", index_col=0).dropna()
display(df.head())
df['EncodedPixels']['000155de5.jpg']
# turn rle example into a list of ints
rle = [int(i) for i in df['EncodedPixels']['000155de5.jpg'].split()]
# turn list of ints into a list of (`start`, `length`) `pairs`
pairs = list(zip(rle[0:-1:2], rle[1::2])) 
pairs[:3]
start = pairs[0][0]
print(f"Original start position: {start}")

coordinate = (start % 768, start // 768)
print(f"Maps to this coordinate: {coordinate}")

back = 768 * coordinate[1] + coordinate[0]
print(f"And back: {back}")
pixels = [(pixel_position % 768, pixel_position // 768) 
                            for start, length in pairs 
                            for pixel_position in range(start, start + length)]
pixels[:3]
def rle_to_pixels(rle_code):
    '''
    Transforms a RLE code string into a list of pixels of a (768, 768) canvas
    '''
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768) 
                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 
                 for pixel_position in range(start, start + length)]
    return pixels

# First three pixels of this particular bounding box:
rle_to_pixels(df['EncodedPixels']['000155de5.jpg'])[0:3]
# Create a matrix of shape (768, 768) full of zeros
canvas = np.zeros((768, 768))

# numpy arrays can't be indexed by a list of pairs [(x1, y1), (x2, y2)]
# but it can be indexed with a tuple with ([x1, x2,..., xn], [y1, y2... yn])
# tuple(zip(*)) does exactly this map.... 
# ref: https://stackoverflow.com/questions/28491230/indexing-a-numpy-array-with-a-list-of-tuples
canvas[tuple(zip(*pixels))] = 1

plt.imshow(canvas);
canvas = np.zeros((768, 768))
pixels = rle_to_pixels(np.random.choice(df['EncodedPixels']))
canvas[tuple(zip(*pixels))] = 1
plt.imshow(canvas);
# An image may have more than one row in the df, 
# Meaning that the image has more than one ship present
# Here we merge those n-ships into the a continuos rle-code for the image....
df = df.groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

load_img = lambda filename: np.array(PIL.Image.open(f"../input/train_v2/{filename}"))

def apply_mask(image, mask):
    for x, y in mask:
        image[x, y, [0, 1]] = 255
    return image

img = load_img(df.loc[0, 'ImageId'])
mask_pixels = rle_to_pixels(df.loc[0, 'EncodedPixels'])
img = apply_mask(img, mask_pixels)
plt.imshow(img);
w = 6
h = 6

_, axes_list = plt.subplots(h, w, figsize=(2*w, 2*h))

for axes in axes_list:
    for ax in axes:
        ax.axis('off')
        row_index = np.random.randint(len(df)) # take a random row from the df
        ax.imshow(apply_mask(load_img(df.loc[row_index, 'ImageId']), rle_to_pixels(df.loc[row_index, 'EncodedPixels'])))
        ax.set_title(df.loc[row_index, 'ImageId'])
