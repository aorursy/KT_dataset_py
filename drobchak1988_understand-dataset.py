import os

import pandas as pd

from skimage.io import imread

from matplotlib import pyplot as plt

import seaborn as sns
def display(display_list):

    plt.figure(figsize=(10, 10))



    for i in range(len(display_list)):

        plt.subplot(1, len(display_list), i+1)

        plt.imshow(display_list[i])

        plt.axis('off')

    plt.show()

def masks_as_image(in_mask_list):

    '''

    Take the individual ship masks and create a single mask array for all ships

    '''

    all_masks = np.zeros(IMG_SIZE, dtype = np.uint8)

    for mask in in_mask_list:

        if isinstance(mask, str):

            all_masks |= rle_decode(mask)

    return all_masks



def rle_decode(mask_rle, shape=(768, 768)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction



# https://github.com/ternaus/TernausNet/blob/master/Example.ipynb

def mask_overlay(image, mask):

    """

    Helper function to visualize mask

    """

    mask = mask.astype(np.uint8)

    weighted_sum = cv2.addWeighted(mask, 0.75, image, 0.5, 0.)

    img = image.copy()

    ind = mask[:, :, 1] > 0    

    img[ind] = weighted_sum[ind]    

    return img
DATA_PATH   = '../input/airbus-ship-detection/'

TRAIN_PATH  = DATA_PATH+'train_v2/'

TEST_PATH   = DATA_PATH+'test_v2/'
train = os.listdir(TRAIN_PATH)

test = os.listdir(TEST_PATH)

print(len(train), len(test))
images = []

for ImageId in train[43:46]:

    image = imread(TRAIN_PATH+ImageId)

    images += [image]

display(images)
df = pd.read_csv(DATA_PATH+'train_ship_segmentations_v2.csv')
df = df.reset_index()

df['ship_count'] = df.groupby('ImageId')['ImageId'].transform('count')

df.loc[df['EncodedPixels'].isnull().values,'ship_count'] = 0  #see infocusp's comment



sns.set_style("white")

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

sns.distplot(df['ship_count'],kde=False)

plt.title('Ship Count Distribution in Train Set')



print(df['ship_count'].describe())
plt.bar(

    ['Ships', 'No Ships'], 

    [len(df[~df.EncodedPixels.isna()].ImageId.unique()),

    len(df[df.EncodedPixels.isna()].ImageId.unique())]);

plt.ylabel('Number of Images');
# This function transforms EncodedPixels into a list of pixels

# Check our previous notebook for a detailed explanation:

# https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes

def rle_to_pixels(rle_code):

    rle_code = [int(i) for i in rle_code.split()]

    pixels = [(pixel_position % 768, pixel_position // 768) 

                 for start, length in list(zip(rle_code[0:-1:2], rle_code[1:-2:2])) 

                 for pixel_position in range(start, start + length)]

    return pixels



def show_pixels_distribution(df):

    """

    Prints the amount of ship and no-ship pixels in the df

    """

    # Total images in the df

    n_images = df['ImageId'].nunique() 

    

    # Total pixels in the df

    total_pixels = n_images * 768 * 768 



    # Keep only rows with RLE boxes, transform them into list of pixels, sum the lengths of those lists

    ship_pixels = df['EncodedPixels'].dropna().apply(rle_to_pixels).str.len().sum() 



    ratio = ship_pixels / total_pixels

    print(f"Ship: {round(ratio, 3)} ({ship_pixels})")

    print(f"No ship: {round(1 - ratio, 3)} ({total_pixels - ship_pixels})")

df = pd.read_csv(DATA_PATH+"train_ship_segmentations_v2.csv").append(pd.read_csv(DATA_PATH+"sample_submission_v2.csv"))

show_pixels_distribution(df)
show_pixels_distribution(df.dropna())