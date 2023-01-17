import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_colwidth', 1000)
train = dd.read_csv("https://s3.amazonaws.com/google-landmark/metadata/train.csv", dtype={'landmark_id':'uint32'}).persist()
train.head(10)
train_attr = dd.read_csv('https://s3.amazonaws.com/google-landmark/metadata/train_attribution.csv').persist()
train_attr.head(10)
lands = pd.DataFrame(train.landmark_id.value_counts().compute())

lands.reset_index(inplace=True)

lands.columns = ['landmark_id','count']
print("Number of classes {}".format(lands.shape[0]))
print("Total of examples in train set = ",lands['count'].sum())
ax = lands['count'].plot(loglog=True, grid=True)

ax.set(xlabel="Landmarks", ylabel="Count")
NUM_THRESHOLD = 50

top_lands = set(lands[lands['count'] >= NUM_THRESHOLD]['landmark_id'])

print("Number of TOP classes {}".format(len(top_lands)))
new_train = train[train['landmark_id'].isin(top_lands)].compute()

print("Total of examples in subset of train: {}".format(new_train.shape[0]))
# Extract site names from urls

sites = train['url'].compute()

sites = sites.str.split('/').tolist()

sites = set([item[2] for item in sites])

sites
from urllib import request #, error

from PIL import Image, ExifTags

from io import BytesIO
def download_image(url):

    try:

        response = request.urlopen(url)

        image_data = response.read()

    except:

         return 1



    try:

        pil_image = Image.open(BytesIO(image_data))

        if pil_image.info.get('exif', None):

            exif_dict = pil_image._getexif()

        else:

            exif_dict = {}

    except:

        return 1



    try:

        pil_image_rgb = pil_image.convert('RGB')

    except:

        return 1

    

    return pil_image_rgb, exif_dict
images = train['url'].compute().reset_index()
idx = 86666
images['url'][idx]
img, exif = download_image(images['url'][idx])
exif = {

    ExifTags.TAGS[k]: v

    for k, v in exif.items()

    if k in ExifTags.TAGS

}

exif['MakerNote'] = ''
exif
img