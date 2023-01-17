!pip install requests bs4 lxml wget
import requests

from bs4 import BeautifulSoup

from multiprocessing import Pool

from functools import partial

import os

import os.path as pth

import wget

from zipfile import ZipFile
def get_metadata(url, base_path='./'):

    filename = url.split('/')[-1]

    full_filename = pth.join(base_path, filename)

    if pth.exists(full_filename):

        return full_filename, 1

    wget.download(url, out=base_path)

    ### If you can't use wget, you can use below blocked code

    ### But It's much slower than wget...

    # data = requests.get(url).data

    # with open(full_filename, 'wb') as f:

    #     f.write(data)

    return full_filename, 0
target_url = 'https://storage.googleapis.com/openimages/web/download.html'
page = requests.get(target_url).text

soup = BeautifulSoup(page, 'lxml')

main = soup.select_one('div.main')

rows = main.select('div.row')

rows = [row for row in rows 

            if row.select_one('div.col-10') and row.select_one('div.col-2.titlecol')]
base_path = 'metadata/'

os.makedirs(base_path, exist_ok=True)
### !!!Remove!!! ###

### Because of kaggle kernel disk space limit. 

### If you use this script, you must remove below line.

rows = rows[:10] ### this line or cell must be removed!
for row in rows:

    sub_name = row.select_one('div.col-2.titlecol').get_text().strip()

    if sub_name:

        sub_path = pth.join(base_path, sub_name)

        os.makedirs(sub_path, exist_ok=True)

        hrefs = [a.get('href') for a in row.select('a') if a.get('href')]

        hrefs = [href for href in hrefs 

                    if href.endswith('.csv') or href.endswith('.txt')]      

        if hrefs:

            download_func = partial(get_metadata, base_path=sub_path)

            pool = Pool(8)

            for filename, status in pool.imap_unordered(download_func, hrefs):

                if status == 0:

                    print(filename + ' is saved.')

                elif status == 1:

                    print(filename + ' is already exist.')

                else:

                    print('???')

                

                ### !!!Remove!!! ###

                ### Because of kaggle kernel disk space limit. 

                ### If you use this script, you must remove below line.

                os.remove(filename) ### this line must be removed!

            

            pool.close()

            pool.join()
base_path = pth.join('metadata', 'Segmentations')

os.makedirs(base_path, exist_ok=True)

base_url = 'https://storage.googleapis.com/openimages/v5/{}-masks/{}-masks-{}.zip'
sub_list = ['train', 'validation', 'test']
### !!!Remove!!! ###

### Because of kaggle kernel disk space limit. 

### If you use this script, you must remove below line.

sub_list = ['validation', 'test'] ### this line or cell must be removed!
for sub_name in sub_list:

    sub_path = pth.join(base_path, sub_name)

    os.makedirs(sub_path, exist_ok=True)

    urls = [base_url.format(sub_name, sub_name, offset) 

            for offset in list(range(10))+['a','b','c','d','e','f']]

    download_func = partial(get_metadata, base_path=sub_path)

    pool = Pool(8)

    for filename, status in pool.imap_unordered(download_func, urls):

        if status == 0:

            print(filename + ' is saved.')

        elif status == 1:

            print(filename + ' is already exist.')

        else:

            print('???')

    pool.close()

    pool.join()
for sub_name in sub_list:

    sub_path = pth.join(base_path, sub_name)

    zip_filename_list = [filename for filename in os.listdir(sub_path) if filename.endswith('.zip')]

    for filename in zip_filename_list:

        segment_zip_filename = pth.join(sub_path, filename)

        with ZipFile(segment_zip_filename, 'r') as zip_ref:

            zip_ref.extractall(sub_path)

        os.remove(segment_zip_filename)

        print(segment_zip_filename+ ' was extracted')
!ls metadata
!ls metadata/Segmentations/test | head
### !!!Remove!!! ###

### Because of kaggle kernel disk space limit. 

### If you use this script, you must remove below line.

!rm -rf metadata/ ### this line or cell must be removed!