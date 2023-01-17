import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import csv
import cv2

%matplotlib inline
basedir = '../input/train_v2/'
train_df = pd.read_csv("../input/train_ship_segmentations_v2.csv")
train_df.head()
train_df["GotShip"] = 0
train_df.loc[train_df["EncodedPixels"].notnull(), "GotShip"] = 1
# train_df['GotShips'] = np.where(train_df['EncodedPixels'].isnull(), 0, 1)
train_df.head()
print('Number of images without ships in train log: ', train_df.ImageId[train_df['GotShip'] == 0].nunique())

# train_df.to_csv("./dataset/train/ships.csv")
noship = train_df[train_df['GotShip'] == 0]
noship.head()
def show_samples(imagedata, no_of_images, no_of_rows=4, no_of_cols=4):
    i = 0
    ship_sx = random.sample(range(0, len(imagedata)), no_of_images)
    samples = imagedata.iloc[ship_sx]
    fig = plt.figure(1, figsize = (20,20))
    for index, row in samples.iterrows():
        i = i + 1
        image = mpimg.imread(basedir + row['ImageId'])
        img = image.copy()
        rszImg = cv2.resize(img, (200, 200), cv2.INTER_AREA)

        ax = fig.add_subplot(no_of_rows, no_of_cols, i)
        ax.set_title(index)
        ax.imshow(rszImg)
        fig.tight_layout()  
show_samples(noship, 16)
print('Number of images with ships in train log: ', train_df.ImageId[train_df['GotShip'] != 0].size)
print('Number of unique images with ships in train log: ', train_df.ImageId[train_df['GotShip'] != 0].nunique())

# train_df.to_csv("./dataset/train/ships.csv")
ship = train_df[train_df['GotShip'] != 0]
ship.head(10)
show_samples(ship, 15)
x = train_df[train_df["ImageId"] == "000194a2d.jpg"]
x
show_samples(x, 5)
# CHECK THAT NO DUPLICATE ENCODEDPIXELS ARE LISTED
duped_ship = ship.drop_duplicates("EncodedPixels")
print (len(duped_ship))
df1 = pd.DataFrame({'':['Ship', 'No Ship'], 'Image Count':[len(ship), len(noship)]})
df1
df1.plot.bar(x='', y='Image Count', rot=0, color='b', legend=None, title="Ship Count Distribution")
# COUNT THE NUMBER OF DUPLICATES EACH IMAGE HAS
unique_ship = ship['ImageId'].value_counts().reset_index()
unique_ship.columns = ['ImageId', 'NumberOfDuplicates']
unique_ship.head()
# COUNT THE NUMBER OF IMAGES vs NUMBER OF DUPLICATES 
dupeship = unique_ship.groupby('NumberOfDuplicates').count()
dupeship
plt.figure()
df2 = pd.DataFrame(dupeship, columns=['NumberOfDuplicates', 'ImageId'])
ax = df2.plot.bar(color='r', legend=None, title="Ship Duplicates Distribution")
ax.set_xlabel("Number of Duplicates")
ax.set_ylabel("Number of Images")
# SAMPLE
idx = random.sample(range(0, len(ship)), 1)
sx_one = ship.iloc[idx]
encodedpixels = sx_one['EncodedPixels'].values
sx_image = sx_one['ImageId'].values
sx = sx_image[0]
sx_base = basedir + sx
sx_base
sample_data = ship[ship['ImageId'] == sx]
sample_data
unique_ship.NumberOfDuplicates[unique_ship['ImageId'] == sx_image[0]]
# CREATE AN IMAGE MASK
mask = np.zeros((768, 768))

# UNRAVEL MASK INTO ARRAY
mask = mask.ravel()
mask
# CREATE SHIP MASK
def encode_rle(encodedpixels, n=2):
    # SPLIT ENCODED PIXELS STRING
    shipmask = encodedpixels.split()
    # CONVERT LIST TO TUPLES
    shipmask = zip(*[iter(shipmask)]*n)
    # CONVERT STRING TO INT
    rle = [(int(start), int(start) + int(length)) for start, length in shipmask]
    return rle
rle_data = sample_data['EncodedPixels'].apply(encode_rle)
rle_data
def total_mask(rle_data, mask):
    for rle in rle_data:
        for start,end in rle:
            print (start, end)
            mask[start:end] = 1
    mask = mask.reshape(768,768).T
    return mask
mask = total_mask(rle_data, mask)
img_mask = np.dstack((mask, mask, mask))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(img_mask)
# SHOW MASK IMAGE
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
ax.set_title("RLE Masking")
ax.imshow(img_mask)

orig_image = mpimg.imread(sx_base)
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Orig Image")
ax.imshow(orig_image)
fig.tight_layout()
x = range(1200)
fig, ax = plt.subplots(1, figsize = (50,50))
ax.imshow(orig_image, extent=[0, 1200, 0, 1200])
poly = np.ascontiguousarray(mask, dtype=np.uint8)
(flags, contours, h) = cv2.findContours(poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_image = orig_image.copy()
cv2.drawContours(contour_image, contours, -1, (0,255,0), 1)
x = range(1200)
fig, ax = plt.subplots(1, figsize = (50,50))
ax.imshow(contour_image, extent=[0, 1200, 0, 1200])