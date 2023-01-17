import pandas as pd 

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/imgs_train.csv')

print(train.shape)

train.head()
# Sample code for loading an image



from PIL import Image

from pathlib import Path

im_p = Path('../input/all_images/image_moderation_images')

Image.open(im_p/train.images_id.iloc[0])