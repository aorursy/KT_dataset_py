import multiprocessing

import os

from io import BytesIO

from urllib import request

import pandas as pd

import re

import tqdm

from PIL import Image





# set files and dir

CSV_PATH, OUT_DIR = '../input/train.csv', '../input/train'  # recognition challenge

# CSV_PATH, OUT_DIR = '../input/index.csv', '../input/index'  # retrieval challenge

# CSV_PATH, OUT_DIR = '../input/test.csv', '../input/test'  # test data



# preferences

TARGET_SIZE = 128  # image resolution to be stored

IMG_QUALITY = 90  # JPG quality

NUM_WORKERS = 8  # Num of CPUs
pd.read_csv(CSV_PATH).url.apply(lambda x: x.split('/')[-2]).value_counts().head()
def overwrite_urls(df):

    def reso_overwrite(url_tail):

        pattern = 's[0-9]+'

        search_result = re.match(pattern, url_tail)

        if search_result is None:

            return url_tail

        else:

            return 's{}'.format(TARGET_SIZE)



    def join_url(parsed_url, s_reso):

        parsed_url[-2] = s_reso

        return '/'.join(parsed_url)



    parsed_url = df.url.apply(lambda x: x.split('/'))

    train_url_tail = parsed_url.apply(lambda x: x[-2])

    resos = train_url_tail.apply(lambda x: reso_overwrite(x))



    overwritten_df = pd.concat([parsed_url, resos], axis=1)

    overwritten_df.columns = ['url', 's_reso']

    df['url'] = overwritten_df.apply(lambda x: join_url(x['url'], x['s_reso']), axis=1)

    return df





def parse_data(df):

    key_url_list = [line[:2] for line in df.values]

    return key_url_list





def download_image(key_url):

    (key, url) = key_url

    filename = os.path.join(OUT_DIR, '{}.jpg'.format(key))



    if os.path.exists(filename):

        print('Image {} already exists. Skipping download.'.format(filename))

        return 0



    try:

        response = request.urlopen(url)

        image_data = response.read()

    except:

        print('Warning: Could not download image {} from {}'.format(key, url))

        return 1



    try:

        pil_image = Image.open(BytesIO(image_data))

    except:

        print('Warning: Failed to parse image {}'.format(key))

        return 1



    try:

        pil_image = pil_image.convert('RGB')

    except:

        print('Warning: Failed to convert image {} to RGB'.format(key))

        return 1



    try:

        pil_image = pil_image.resize((TARGET_SIZE, TARGET_SIZE))

    except:

        print('Warning: Failed to resize image {}'.format(key))

        return 1



    try:

        pil_image.save(filename, format='JPEG', quality=IMG_QUALITY)

    except:

        print('Warning: Failed to save image {}'.format(filename))

        return 1



    return 0





def loader(df):

    if not os.path.exists(OUT_DIR):

        os.mkdir(OUT_DIR)



    key_url_list = parse_data(df)

    pool = multiprocessing.Pool(processes=NUM_WORKERS)

    failures = sum(tqdm.tqdm(pool.imap_unordered(download_image, key_url_list),

                             total=len(key_url_list)))

    print('Total number of download failures:', failures)

    pool.close()

    pool.terminate()
# now, start downloading

if __name__ == '__main__':

    df = pd.read_csv(CSV_PATH).query('url != "None"')

    loader(overwrite_urls(df))