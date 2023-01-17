# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#reading the data 

train_data = pd.read_csv("/kaggle/input/train.csv")
train_data.head(5)
train_data.isnull().sum()
total_data = train_data.count()
print(total_data)
train_data = train_data.dropna()
train_data.head()
#lets put the image name , in the name , id in the Id and classId in the classId

new_train_df = pd.DataFrame(columns=['ImageId','hasMask', 'ClassId','EncodedPixels'])

new_train_df['ImageId'] = train_data['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

new_train_df['ClassId'] = train_data['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

new_train_df['hasMask'] = ~ train_data['EncodedPixels'].isna()

new_train_df['EncodedPixels'] = train_data['EncodedPixels'] 

mask_count_df = new_train_df.groupby('ImageId').agg(np.sum).reset_index()

mask_count_df.sort_values('hasMask', ascending=False, inplace=True)

print(mask_count_df.shape)

mask_count_df.head()
new_train_df.head()
ax = sns.catplot(x='ClassId',kind='count',data=new_train_df,orient="h")

ax.fig.autofmt_xdate()
sub_df = pd.read_csv('../input/sample_submission.csv')

sub_df['ImageId'] = sub_df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])

test_imgs = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
test_imgs.head()
def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels= img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(mask_rle, shape=(1600,256)):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (width,height) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T
def build_masks(rles, input_shape):

    depth = len(rles)

    height, width = input_shape

    masks = np.zeros((height, width, depth))

    

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, (width, height))

    

    return masks



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
# sample_filename = 'db4867ee8.jpg'

# sample_image_df = new_train_df[new_train_df['ImageId'] == sample_filename]

# sample_path = f"../input/train_images/{sample_image_df['ImageId'].iloc[0]}"

# sample_img = cv2.imread(sample_path)

# sample_rles = sample_image_df['EncodedPixels'].values

# sample_masks = build_masks(sample_rles, input_shape=(256, 1600))



# fig, axs = plt.subplots(5, figsize=(12, 12))

# axs[0].imshow(sample_img)

# axs[0].axis('off')



# for i in range(4):

#     axs[i+1].imshow(sample_masks[:, :, i])

#     axs[i+1].axis('off')
