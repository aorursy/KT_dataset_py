import cv2

import glob

import numpy as np

import pandas as pd 

import pydicom

from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

import os

from PIL import Image
SAMPLE_DATA_DIR = '../input/siim-acr-pneumothorax-segmentation/sample images'

DATA_DIR = '../input/siim-train-test/siim/'

BASE_WIDTH = 1024

IMAGE_ID = '1.2.276.0.7230010.3.1.4.8323329.4904.1517875185.355709'
os.listdir(SAMPLE_DATA_DIR)
ds = pydicom.dcmread(f"{SAMPLE_DATA_DIR}/{IMAGE_ID}.dcm")
print(ds.pixel_array.shape)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
Image.fromarray(ds.pixel_array)
def rle2mask(rle, width, height):

    mask= np.zeros(width * height, dtype=np.uint8)

    array = np.asarray([int(x) for x in rle.split()])

    starts = array[0::2]

    lengths = array[1::2]



    current_position = 0

    for index, start in enumerate(starts):

        current_position += start

        # see https://github.com/tensorflow/models/issues/3906#issuecomment-391998102

        # The segmentation ground truth images in your custom dataset should have

        # 1, 2, 3, ..., num_class grayscale value at each pixel (0 for background).

        # For example if you have 2 classes, you should use 1 and 2 for corresponding pixel.

        # Of course the segmentation mask will look almost "black". If you choose,

        # say 96 and 128, for your segmentation mask to make the it looks more human friendly,

        #the network may end up predicting labels greater than num_class,

        # which leads to the error in this issue.

        mask[current_position:current_position+lengths[index]] = 1  # Do NOT use 255

        current_position += lengths[index]



    return mask.reshape(width, height)
df = pd.read_csv(os.path.join(SAMPLE_DATA_DIR, 'train-rle-sample.csv'), header=None, names=['ImageId', 'EncodedPixels'])
df.head()
df[df['ImageId'] == IMAGE_ID]
rle: str = df[df['ImageId'] == IMAGE_ID]['EncodedPixels'].values[0]
image_bytes = rle2mask(rle, BASE_WIDTH, BASE_WIDTH)
Image.fromarray(image_bytes.T * 255)
fig,ax = plt.subplots(1)

ax.imshow(ds.pixel_array, cmap=plt.cm.bone)

ax.imshow(image_bytes.T, alpha=0.5)

plt.show()
def dcm_to_png(dcm_file: str, png_file: str, width=BASE_WIDTH):

    assert os.path.exists(dcm_file)

    assert dcm_file.endswith('.dcm')

    assert png_file.endswith('.png')

    ds = pydicom.dcmread(dcm_file)

    img_bytes = ds.pixel_array if width == BASE_WIDTH else cv2.resize(ds.pixel_array, (width, width))

    res, im_png = cv2.imencode('.png', img_bytes)

    assert res == True

    with open(png_file, 'wb') as f:

        f.write(im_png.tobytes())
dcm_to_png(

    os.path.join(SAMPLE_DATA_DIR, IMAGE_ID+'.dcm'),

    f'{IMAGE_ID}.png',

)
img = cv2.imread(f'{IMAGE_ID}.png', cv2.IMREAD_GRAYSCALE)
assert img.shape == (BASE_WIDTH, BASE_WIDTH)
assert np.array_equal(img, pydicom.dcmread(os.path.join(SAMPLE_DATA_DIR, IMAGE_ID+'.dcm')).pixel_array)
Image.fromarray(img)
os.remove(f'{IMAGE_ID}.png')
def mask_to_png(rle: str, png_file: str, width=BASE_WIDTH):

    assert rle

    assert png_file.endswith('.png')

    img = rle2mask(rle, width, width) if rle != '-1' else np.zeros((width, width), dtype=np.uint8)

    res, img_png = cv2.imencode('.png', img.T)

    assert res == True

    with open(png_file, 'wb') as f:

        f.write(img_png)
mask_to_png(rle, 'mask.png')
img = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
assert img.shape == (BASE_WIDTH, BASE_WIDTH)
Image.fromarray(img * 255)
os.remove('mask.png')
def dcm_to_png_dir(input_dir: str, output_dir: str, width=BASE_WIDTH):

    assert os.path.exists(input_dir)

    assert not os.path.exists(output_dir)

    os.makedirs(output_dir)



    dcm_files = glob.glob(f'{input_dir}/**/*.dcm', recursive=True)

    

    for dcm_file in tqdm(dcm_files, desc=f'{os.path.basename(output_dir)}'):

        image_id = os.path.basename(dcm_file)[0: -len('.dcm')]

        dcm_to_png(dcm_file, os.path.join(output_dir, image_id + '.png'), width)
dcm_to_png_dir(SAMPLE_DATA_DIR, '../data/preprocessed/sample-images-128x128', 128)
dcm_to_png_dir(f'{DATA_DIR}/dicom-images-test', '../data/preprocessed/128x128/test', 128)
dcm_to_png_dir(f'{DATA_DIR}/dicom-images-train', '../data/preprocessed/128x128/train', 128)
def calc_mask(grouped: pd.core.groupby.DataFrameGroupBy, image_id: str, width:int)->np.ndarray:

    df = grouped.get_group(image_id)

    result = []

    for _, row in df.iterrows():

        rle = row['EncodedPixels'].strip()

        if rle == '-1':

            mask = np.zeros((width, width), dtype=np.uint8)

        else:

            mask = rle2mask(rle, width, width)

        result.append(mask)



    assert len(result) == len(df)

    if len(df) > 1:

        mask = np.array(result).sum(0).astype(np.uint8)

    else:

        mask = result[0]

    return mask.T



def mask_to_png_dir(rle_df: pd.DataFrame, output_dir: str, width=BASE_WIDTH):

    assert not os.path.exists(output_dir)

    os.makedirs(output_dir)



    rle_df['EncodedPixels'] = rle_df['EncodedPixels'].astype(str)

    grouped = rle_df.groupby('ImageId')

    for image_id in tqdm(grouped.groups.keys(), desc=f'{os.path.basename(output_dir)}'):

        mask = calc_mask(grouped, image_id, width)

        res, img_png = cv2.imencode('.png', mask)

        assert res == True



        with open(os.path.join(output_dir, image_id+'.png'), 'wb') as f:

            f.write(img_png)
mask_to_png_dir(

    pd.read_csv(f'{SAMPLE_DATA_DIR}/train-rle-sample.csv', header=None, names=['ImageId', 'EncodedPixels']),

    '../data/preprocessed/128x128/sample-masks',

    128,

)
mask_to_png_dir(

    pd.read_csv(f'{DATA_DIR}/train-rle.csv', skiprows=1, header=None, names=['ImageId', 'EncodedPixels']),

    '../data/preprocessed/128x128/masks',

    128,

)
def missing_masks(train_masks_dir: str, train_images_dir: str, width: int):

    train_images = [os.path.basename(file_path)[0: -len('.dcm')] for file_path in glob.glob(f'{train_images_dir}/**/*.png', recursive=True)]

    train_masks = [os.path.basename(file_path)[0: -len('.dcm')] for file_path in glob.glob(f'{train_masks_dir}/*.png', recursive=True)]

    missing_masks = set(train_images) - set(train_masks)

    for image_id in tqdm(missing_masks, desc='Missing Masks'):

        mask = np.zeros((width, width), dtype=np.uint8)

        res, img_png = cv2.imencode('.png', mask)

        assert res == True



        with open(os.path.join(train_masks_dir, image_id+'.png'), 'wb') as f:

            f.write(img_png)
missing_masks(

    '../data/preprocessed/128x128/masks',

    '../data/preprocessed/128x128/train',

    128,

)
for width in tqdm([256, 512, 1024]):

    dcm_to_png_dir(f'{DATA_DIR}/dicom-images-test', f'../data/preprocessed/{width}x{width}/test', width)

    dcm_to_png_dir(f'{DATA_DIR}/dicom-images-train', f'../data/preprocessed/{width}x{width}/train', width)

    mask_to_png_dir(

        pd.read_csv(f'{DATA_DIR}/train-rle.csv', skiprows=1, header=None, names=['ImageId', 'EncodedPixels']),

        f'../data/preprocessed/{width}x{width}/masks',

        width,

    )

    missing_masks(

        f'../data/preprocessed/{width}x{width}/masks',

        f'../data/preprocessed/{width}x{width}/train',

        width,

    )